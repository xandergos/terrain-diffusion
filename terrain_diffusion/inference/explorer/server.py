import io
import os
from typing import Optional

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from flask import Flask, Response, jsonify, request, send_from_directory

from terrain_diffusion.inference.world_pipeline import WorldPipeline, resolve_hdf5_path
from terrain_diffusion.inference.relief_map import get_relief_map
from terrain_diffusion.common.cli_helpers import parse_kwargs, parse_cache_size

app = Flask(__name__, static_folder='static')

_PIPELINE: Optional[WorldPipeline] = None
_PIPELINE_CONFIG: dict = {}

CHANNEL_NAMES = ['Elev', 'p5', 'Temp', 'T std', 'Precip', 'Precip CV']


def _select_device() -> str:
    dev = os.getenv("TERRAIN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cpu":
        print("Warning: Using CPU (CUDA not available).")
    return dev


def _get_pipeline() -> WorldPipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    cfg = _PIPELINE_CONFIG
    caching_strategy = cfg.get('caching_strategy', 'indirect')
    _PIPELINE = WorldPipeline.from_pretrained(
        cfg.get('model_path', 'xandergos/terrain-diffusion-90m'),
        seed=cfg.get('seed'),
        latents_batch_size=cfg.get('latents_batch_size', [1, 2, 4, 8, 16]),
        log_mode=cfg.get('log_mode', 'verbose'),
        torch_compile=cfg.get('torch_compile', False),
        dtype=cfg.get('dtype'),
        caching_strategy=caching_strategy,
        **cfg.get('kwargs', {}),
    )
    _PIPELINE.to(cfg.get('device') or _select_device())
    hdf5_file = cfg.get('hdf5_file')
    if caching_strategy == 'direct':
        _PIPELINE.bind(hdf5_file=hdf5_file)
    else:
        _PIPELINE.bind(hdf5_file or 'TEMP')
    print(f"World seed: {_PIPELINE.seed}")
    return _PIPELINE


def _coarse_channel(world: WorldPipeline, ci0: int, ci1: int, cj0: int, cj1: int, channel: int) -> np.ndarray:
    """Return channel data in real units. ch 0/1: signed-sqrt -> metres."""
    coarse = world.coarse[:, ci0:ci1, cj0:cj1]
    data = (coarse[:-1] / (coarse[-1:] + 1e-8))[channel].detach().cpu().numpy()
    if channel <= 1:
        data = np.sign(data) * np.square(data)
    return data


def _png_response(rgba: np.ndarray) -> Response:
    buf = io.BytesIO()
    plt.imsave(buf, np.clip(rgba, 0, 1), format='png')
    buf.seek(0)
    return Response(buf.read(), mimetype='image/png')


@app.get('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.get('/api/status')
def status():
    world = _get_pipeline()
    return jsonify({'seed': str(world.seed), 'channels': CHANNEL_NAMES, 'native_resolution': world.native_resolution})


@app.post('/api/seed')
def set_seed():
    data = request.get_json()
    if 'seed' not in data:
        return jsonify({'error': 'seed required'}), 400
    world = _get_pipeline()
    world.change_seed(int(data['seed']))
    return jsonify({'seed': str(world.seed)})


@app.post('/api/new_seed')
def new_seed():
    world = _get_pipeline()
    world.change_seed()
    return jsonify({'seed': str(world.seed)})


@app.get('/api/coarse.png')
def coarse_png():
    try:
        world = _get_pipeline()
        channel = request.args.get('channel', 0, int)
        ci0 = request.args.get('ci0', -50, int)
        ci1 = request.args.get('ci1', 50, int)
        cj0 = request.args.get('cj0', -50, int)
        cj1 = request.args.get('cj1', 50, int)

        data = _coarse_channel(world, ci0, ci1, cj0, cj1, channel)
        display = np.log1p(np.maximum(data, 0)) if channel == 4 else data
        vmin, vmax = float(np.nanmin(display)), float(np.nanmax(display))
        if vmax == vmin:
            vmax = vmin + 1
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        rgba = plt.get_cmap('viridis')(norm(display)).astype(np.float32)

        # Channels available for filtering (skip p5 = channel 1)
        filterable = [0, 2, 3, 4, 5]
        filter_active = any(
            request.args.get(f'ch{ch}_{side}', type=float) is not None
            for ch in filterable for side in ('min', 'max')
        )
        if filter_active:
            mask = np.ones(data.shape, dtype=bool)
            for ch in filterable:
                lo = request.args.get(f'ch{ch}_min', type=float)
                hi = request.args.get(f'ch{ch}_max', type=float)
                if lo is not None or hi is not None:
                    ch_data = _coarse_channel(world, ci0, ci1, cj0, cj1, ch)
                    if lo is not None:
                        mask &= ch_data >= lo
                    if hi is not None:
                        mask &= ch_data <= hi
            rgba[~mask, :3] *= 0.3

        resp = _png_response(rgba)
        resp.headers['X-Vmin'] = str(round(vmin, 3))
        resp.headers['X-Vmax'] = str(round(vmax, 3))
        resp.headers['Access-Control-Expose-Headers'] = 'X-Vmin, X-Vmax'
        return resp
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400


@app.get('/api/coarse_data.json')
def coarse_data():
    """Return all channel values as 2-D arrays for client-side hover lookup."""
    try:
        world = _get_pipeline()
        ci0 = request.args.get('ci0', -50, int)
        ci1 = request.args.get('ci1', 50, int)
        cj0 = request.args.get('cj0', -50, int)
        cj1 = request.args.get('cj1', 50, int)
        channels = {
            name: np.round(_coarse_channel(world, ci0, ci1, cj0, cj1, i), 2).tolist()
            for i, name in enumerate(CHANNEL_NAMES)
        }
        return jsonify({'ci0': ci0, 'ci1': ci1, 'cj0': cj0, 'cj1': cj1, 'channels': channels})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.get('/api/coarse_stats')
def coarse_stats():
    try:
        world = _get_pipeline()
        ci0 = request.args.get('ci0', -50, int)
        ci1 = request.args.get('ci1', 50, int)
        cj0 = request.args.get('cj0', -50, int)
        cj1 = request.args.get('cj1', 50, int)
        stats = {}
        for ch in range(len(CHANNEL_NAMES)):
            data = _coarse_channel(world, ci0, ci1, cj0, cj1, ch)
            stats[ch] = {
                'name': CHANNEL_NAMES[ch],
                'min': round(float(np.nanmin(data)), 3),
                'max': round(float(np.nanmax(data)), 3),
            }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.get('/api/detail.png')
def detail_png():
    try:
        world = _get_pipeline()
        ci = request.args.get('ci', 0, int)
        cj = request.args.get('cj', 0, int)
        detail_size = request.args.get('detail_size', 1024, int)
        pan_i = request.args.get('pan_i', 0, int)
        pan_j = request.args.get('pan_j', 0, int)
        mode = request.args.get('mode', 'relief')

        center_i = ci * 256 + pan_i
        center_j = cj * 256 + pan_j
        half = detail_size // 2

        region = world.get(center_i - half, center_j - half, center_i + half, center_j + half)
        elev = region['elev'].cpu().numpy()

        if mode == 'elevation':
            vmin, vmax = float(np.nanmin(elev)), float(np.nanmax(elev))
            if vmax == vmin:
                vmax = vmin + 1
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            rgba = plt.get_cmap('terrain')(norm(elev)).astype(np.float32)
        elif mode == 'temperature' and region.get('climate') is not None:
            temp = region['climate'][0].cpu().numpy()
            vmin, vmax = float(np.nanmin(temp)), float(np.nanmax(temp))
            if vmax == vmin:
                vmax = vmin + 1
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            rgba = plt.get_cmap('RdBu_r')(norm(temp)).astype(np.float32)
        else:
            relief_rgb = get_relief_map(elev, None, None, None)
            rgba = np.concatenate([
                np.clip(relief_rgb, 0, 1),
                np.ones((*relief_rgb.shape[:2], 1), dtype=np.float32),
            ], axis=-1)

        return _png_response(rgba)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400


@app.get('/api/detail_raw')
def detail_raw():
    """
    Binary: int16-LE elevation (H*W*2 bytes) + float32-LE temperature (H*W*4 bytes).
    Headers: X-Height, X-Width, X-Has-Temp.
    """
    try:
        world = _get_pipeline()
        ci = request.args.get('ci', 0, int)
        cj = request.args.get('cj', 0, int)
        detail_size = request.args.get('detail_size', 1024, int)
        pan_i = request.args.get('pan_i', 0, int)
        pan_j = request.args.get('pan_j', 0, int)

        center_i = ci * 256 + pan_i
        center_j = cj * 256 + pan_j
        half = detail_size // 2

        region = world.get(center_i - half, center_j - half, center_i + half, center_j + half)
        elev_np = region['elev'].cpu().numpy().astype(np.float32)
        elev_i16 = np.clip(np.floor(elev_np), -32768, 32767).astype('<i2')
        h, w = elev_i16.shape

        payload = elev_i16.tobytes()
        has_temp = region.get('climate') is not None
        if has_temp:
            payload += region['climate'][0].cpu().numpy().astype('<f4').tobytes()

        resp = Response(payload, mimetype='application/octet-stream')
        resp.headers['X-Height'] = str(h)
        resp.headers['X-Width'] = str(w)
        resp.headers['X-Has-Temp'] = '1' if has_temp else '0'
        resp.headers['Access-Control-Expose-Headers'] = 'X-Height, X-Width, X-Has-Temp'
        return resp
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400


@click.command()
@click.argument("model_path", default="xandergos/terrain-diffusion-90m")
@click.option("--caching-strategy", type=click.Choice(["indirect", "direct"]), default="direct")
@click.option("--hdf5-file", default=None)
@click.option("--cache-size", default="100M")
@click.option("--seed", type=int, default=None)
@click.option("--device", default=None)
@click.option("--batch-size", default="1,2,4,8,16")
@click.option("--log-mode", type=click.Choice(["info", "verbose"]), default="verbose")
@click.option("--compile/--no-compile", "torch_compile", default=True)
@click.option("--dtype", type=click.Choice(["fp32", "bf16", "fp16"]), default="fp32")
@click.option("--host", default="0.0.0.0")
@click.option("--port", type=int, default=int(os.getenv("PORT", "8080")))
@click.option("--kwarg", "extra_kwargs", multiple=True)
def main(model_path, hdf5_file, caching_strategy, cache_size, seed, device,
         batch_size, log_mode, torch_compile, dtype, host, port, extra_kwargs):
    """Terrain Explorer web app"""
    global _PIPELINE_CONFIG
    if caching_strategy == 'indirect' and hdf5_file is None:
        hdf5_file = 'TEMP'
    if hdf5_file is not None:
        hdf5_file = resolve_hdf5_path(hdf5_file)
    batch_sizes = [int(x) for x in batch_size.split(',')] if ',' in batch_size else int(batch_size)
    if dtype == 'fp32':
        dtype = None
    _PIPELINE_CONFIG = {
        'model_path': model_path,
        'hdf5_file': hdf5_file,
        'caching_strategy': caching_strategy,
        'cache_limit': parse_cache_size(cache_size),
        'seed': seed,
        'device': device,
        'latents_batch_size': batch_sizes,
        'log_mode': log_mode,
        'torch_compile': torch_compile,
        'dtype': dtype,
        'kwargs': parse_kwargs(extra_kwargs),
    }
    _get_pipeline()
    app.run(host=host, port=port, debug=False, threaded=False)


if __name__ == '__main__':
    main()
