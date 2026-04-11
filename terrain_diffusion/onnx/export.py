"""Export WorldPipeline EDMUnet2D models to ONNX.

Usage:
    python -m terrain_diffusion.onnx.export xandergos/terrain-diffusion-30m
    python -m terrain_diffusion.onnx.export ./terrain-diffusion-30m --output ./onnx_out --verify
    python -m terrain_diffusion.onnx.export xandergos/terrain-diffusion-30m --models base_model decoder_model

Requires: pip install onnx  (and onnxruntime for --verify)
"""
import functools
import os
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.nn.functional as F

from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.inference.world_pipeline import WorldPipeline
from terrain_diffusion.models.unet_block import UNetBlock


_MODEL_SUBFOLDERS = {
    "coarse_model": WorldPipeline.COARSE_MODEL_FOLDER,
    "base_model":   WorldPipeline.BASE_MODEL_FOLDER,
    "decoder_model": WorldPipeline.DECODER_MODEL_FOLDER,
}


def _resample_onnx(x, mode: str = "keep", factor: int = 2):
    """ONNX-friendly drop-in for mp_layers.resample.

    The original creates a convolution kernel from x.shape[1] at runtime, which
    breaks ONNX shape inference after a Clip node.  These implementations are
    numerically equivalent but use only static-shape operations:
      - down  → stride-2 slice  (same as conv with 1×1 all-ones kernel, stride 2)
      - up    → repeat_interleave (same as transposed conv with 2×2 all-ones, stride 2)
    """
    if mode == "keep":
        return x
    if mode == "down":
        return x[:, :, ::factor, ::factor]
    if mode == "up":
        return x.repeat_interleave(factor, dim=2).repeat_interleave(factor, dim=3)
    if mode == "up_bilinear":
        return F.interpolate(x, scale_factor=float(factor), mode="bilinear", align_corners=False)
    raise ValueError(f"Unknown resample mode: {mode!r}")


def _patch_resample(model: nn.Module) -> dict:
    """Replace UNetBlock.resample partials with ONNX-friendly equivalents.

    Returns a dict of {module: original_resample} so the caller can restore them.
    """
    saved = {}
    for module in model.modules():
        if isinstance(module, UNetBlock) and not isinstance(module.resample, nn.Module):
            saved[module] = module.resample
            mode = module.resample.keywords.get("mode", "keep")
            factor = module.resample.keywords.get("factor", 2)
            module.resample = functools.partial(_resample_onnx, mode=mode, factor=factor)
    return saved


def _restore_resample(saved: dict) -> None:
    for module, original in saved.items():
        module.resample = original


class _PatchConv2dPadding:
    """Context manager that normalises 1-element padding tuples in F.conv2d.

    MPConv passes padding=(k//2,) which PyTorch accepts but the ONNX exporter
    translates to pads=[p, p] (length 2) instead of the required [p, p, p, p]
    (length 4), causing OnnxRuntime to reject the model.  Expanding to an int
    makes the exporter produce the correct 4-element pads attribute.
    """
    def __enter__(self):
        self._orig = F.conv2d

        def _fixed(input, weight, bias=None, stride=1, padding=0,
                   dilation=1, groups=1):
            if isinstance(padding, (tuple, list)) and len(padding) == 1:
                padding = padding[0]
            return self._orig(input, weight, bias, stride, padding, dilation, groups)

        F.conv2d = _fixed
        # Also patch the reference inside torch.nn.functional used by nn.Conv2d internals
        import torch.nn.functional as _F
        _F.conv2d = _fixed
        return self

    def __exit__(self, *_):
        F.conv2d = self._orig
        import torch.nn.functional as _F
        _F.conv2d = self._orig


class _CondWrapper(nn.Module):
    """Flat-arg wrapper around EDMUnet2D for ONNX tracing.

    ONNX tracing cannot represent a Python list as an input, so conditional
    inputs are accepted as individual positional arguments and reassembled
    into a list before being forwarded to the model.
    """
    def __init__(self, model: EDMUnet2D):
        super().__init__()
        self.model = model

    def forward(self, x, noise_labels, *cond_inputs):
        return self.model(x, noise_labels, list(cond_inputs))


def _dummy_inputs(model: EDMUnet2D, batch_size: int, device: str, image_size: int | None = None) -> tuple:
    cfg = model.config
    size = image_size if image_size is not None else cfg.image_size
    x = torch.randn(batch_size, cfg.in_channels, size, size, device=device)
    noise_labels = torch.randn(batch_size, device=device)

    conds = []
    for type_, dim, _ in (cfg.conditional_inputs or []):
        if type_ == "float":
            conds.append(torch.randn(batch_size, device=device))
        elif type_ == "tensor":
            conds.append(torch.randn(batch_size, dim, device=device))
        elif type_ == "embedding":
            conds.append(torch.zeros(batch_size, dtype=torch.long, device=device))

    return (x, noise_labels, *conds)


def export_model(
    model: EDMUnet2D,
    output_path: str | Path,
    *,
    device: str = "cpu",
    opset: int = 17,
    image_size: int | None = None,
) -> None:
    """Export a single EDMUnet2D model to ONNX.

    Args:
        model: Model to export (placed in eval mode automatically).
        output_path: Destination .onnx file path.
        device: Device used during tracing.
        opset: ONNX opset version.
    """
    model = model.eval().to(device)
    n_cond = len(model.config.conditional_inputs or [])

    dummy = _dummy_inputs(model, batch_size=1, device=device, image_size=image_size)
    input_names  = ["x", "noise_labels"] + [f"cond_{i}" for i in range(n_cond)]
    output_names = ["output"]
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    saved = _patch_resample(model)
    try:
        with _PatchConv2dPadding(), torch.no_grad():
            torch.onnx.export(
                _CondWrapper(model),
                dummy,
                str(output_path),
                dynamo=False,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset,
            )
    finally:
        _restore_resample(saved)
    print(f"  Exported → {output_path}")


def verify_model(
    output_path: str | Path,
    model: EDMUnet2D,
    *,
    device: str = "cpu",
    image_size: int | None = None,
) -> None:
    """Compare ONNX and PyTorch outputs; prints the max absolute difference.

    Requires onnxruntime: pip install onnxruntime
    """
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("  [skip] onnxruntime not installed; run: pip install onnxruntime")
        return

    model = model.eval().to(device)
    dummy = _dummy_inputs(model, batch_size=2, device=device, image_size=image_size)

    with torch.no_grad():
        torch_out = model(dummy[0], dummy[1], list(dummy[2:])).cpu().numpy()

    sess = ort.InferenceSession(str(output_path))
    feed = {inp.name: t.cpu().numpy() for inp, t in zip(sess.get_inputs(), dummy)}
    ort_out = sess.run(None, feed)[0]

    max_diff = np.abs(torch_out - ort_out).max()
    print(f"  Verified  max |diff| = {max_diff:.2e}")


@click.command("onnx-export")
@click.argument("model_path")
@click.option("--output", "-o", default="onnx_export", show_default=True,
              help="Output directory for .onnx files.")
@click.option("--device", default="cpu", show_default=True,
              help="Device for tracing, e.g. cpu or cuda.")
@click.option("--opset", default=17, show_default=True,
              help="ONNX opset version.")
@click.option("--verify", is_flag=True,
              help="Verify exported models against PyTorch (requires onnxruntime).")
@click.option("--models", "-m", multiple=True,
              type=click.Choice(list(_MODEL_SUBFOLDERS)), default=list(_MODEL_SUBFOLDERS),
              help="Sub-models to export (repeatable, default: all three).")
def main(model_path, output, device, opset, verify, models):
    """Export WorldPipeline EDMUnet2D models to ONNX.

    MODEL_PATH is a local directory or HuggingFace repo ID,
    e.g. ./terrain-diffusion-30m or xandergos/terrain-diffusion-30m.
    """
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in models:
        subfolder = _MODEL_SUBFOLDERS[name]
        click.echo(f"Loading {name} ({model_path}/{subfolder}) ...")
        try:
            model = EDMUnet2D.from_pretrained(model_path, subfolder=subfolder)
        except Exception as exc:
            click.echo(f"  [skip] {exc}")
            continue

        click.echo(f"Exporting {name} ...")
        out_path = output_dir / f"{name}.onnx"
        size_override = 64 if name in ("coarse_model", "base_model") else None
        export_model(model, out_path, device=device, opset=opset, image_size=size_override)

        if verify:
            click.echo(f"Verifying {name} ...")
            verify_model(out_path, model, device=device, image_size=size_override)


if __name__ == "__main__":
    main()
