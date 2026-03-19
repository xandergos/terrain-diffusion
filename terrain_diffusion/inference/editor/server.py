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
from rasterio.io import MemoryFile

from terrain_diffusion.common.cli_helpers import parse_cache_size, parse_kwargs
from terrain_diffusion.inference.relief_map import get_relief_map
from terrain_diffusion.inference.world_pipeline import WorldPipeline, resolve_hdf5_path

app = Flask(__name__, static_folder='static')

_PIPELINE: Optional[WorldPipeline] = None
_PIPELINE_CONFIG: dict = {}
EDIT_CHANNELS = ['Elev', 'Temp', 'T std', 'Precip', 'P CV']
# Per-channel multiplier applied before sending to the user; inverse applied on input.
# T std is stored as °C×100 (WorldClim Bio4 convention) but displayed/edited in °C.
CHANNEL_DISPLAY_SCALE = {2: 0.01}


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
        cfg.get('model_path', 'xandergos/terrain-diffusion-30m'),
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


def _conditioning_channel(world: WorldPipeline, ci0: int, ci1: int, cj0: int, cj1: int, channel: int, finalized_temp: bool = False) -> np.ndarray:
    if finalized_temp and channel == 1:
        data = world.get_conditioning_preview_finalized(ci0, ci1, cj0, cj1)[channel]
    else:
        data = world.get_conditioning_preview(ci0, ci1, cj0, cj1)[channel]
    return _to_display_units(channel, data)


def _refined_channel(world: WorldPipeline, ci0: int, ci1: int, cj0: int, cj1: int, channel: int) -> np.ndarray:
    return _to_display_units(channel, world.get_refined_preview(ci0, ci1, cj0, cj1)[channel])


def _to_display_units(channel: int, value):
    scale = CHANNEL_DISPLAY_SCALE.get(channel, 1.0)
    return value * scale if scale != 1.0 else value


def _from_display_units(channel: int, value):
    scale = CHANNEL_DISPLAY_SCALE.get(channel, 1.0)
    return value / scale if scale != 1.0 else value


def _png_response(rgba: np.ndarray) -> Response:
    buf = io.BytesIO()
    plt.imsave(buf, np.clip(rgba, 0, 1), format='png')
    buf.seek(0)
    return Response(buf.read(), mimetype='image/png')


def _read_tiff_elevation(file_storage) -> np.ndarray:
    raw = file_storage.read()
    if not raw:
        raise ValueError("Uploaded file is empty.")
    with MemoryFile(raw) as memfile:
        with memfile.open() as dataset:
            elev = dataset.read(1).astype(np.float32)
            nodata = dataset.nodata
    if nodata is not None:
        elev = np.where(elev == nodata, -1000.0, elev)
    elev = np.where(np.isfinite(elev), elev, -1000.0)
    return elev


def _rebuild_pipeline(world: WorldPipeline):
    world.rebuild()
    return world


def _channel_cmap(channel: int) -> str:
    if channel == 0:
        return 'terrain'
    if channel == 1:
        return 'RdBu_r'
    if channel == 2:
        return 'magma'
    if channel == 3:
        return 'Blues'
    return 'viridis'


@app.get('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.get('/api/status')
def status():
    world = _get_pipeline()
    return jsonify({
        'seed': str(world.seed),
        'native_resolution': world.native_resolution,
        'channels': EDIT_CHANNELS,
        'cond_snr': list(world.kwargs['cond_snr']),
        'brush_modes': ['set', 'raise', 'lower', 'smooth'],
    })


@app.post('/api/seed')
def set_seed():
    data = request.get_json() or {}
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
        finalized_temp = bool(request.args.get('finalized_temp', 0, int))

        data = _conditioning_channel(world, ci0, ci1, cj0, cj1, channel, finalized_temp=finalized_temp)
        vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
        if vmax == vmin:
            vmax = vmin + 1
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        rgba = plt.get_cmap(_channel_cmap(channel))(norm(data)).astype(np.float32)

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
    try:
        world = _get_pipeline()
        ci0 = request.args.get('ci0', -50, int)
        ci1 = request.args.get('ci1', 50, int)
        cj0 = request.args.get('cj0', -50, int)
        cj1 = request.args.get('cj1', 50, int)
        finalized_temp = bool(request.args.get('finalized_temp', 0, int))
        preview = world.get_conditioning_preview(ci0, ci1, cj0, cj1)
        if finalized_temp:
            preview[1] = world.get_conditioning_preview_finalized(ci0, ci1, cj0, cj1)[1]
        channels = {
            name: np.round(_to_display_units(i, preview[i]), 2).tolist()
            for i, name in enumerate(EDIT_CHANNELS)
        }
        return jsonify({'ci0': ci0, 'ci1': ci1, 'cj0': cj0, 'cj1': cj1, 'channels': channels})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.get('/api/refined.png')
def refined_png():
    try:
        world = _get_pipeline()
        channel = request.args.get('channel', 0, int)
        ci0 = request.args.get('ci0', -50, int)
        ci1 = request.args.get('ci1', 50, int)
        cj0 = request.args.get('cj0', -50, int)
        cj1 = request.args.get('cj1', 50, int)

        data = _refined_channel(world, ci0, ci1, cj0, cj1, channel)
        vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
        if vmax == vmin:
            vmax = vmin + 1
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        rgba = plt.get_cmap(_channel_cmap(channel))(norm(data)).astype(np.float32)

        resp = _png_response(rgba)
        resp.headers['X-Vmin'] = str(round(vmin, 3))
        resp.headers['X-Vmax'] = str(round(vmax, 3))
        resp.headers['Access-Control-Expose-Headers'] = 'X-Vmin, X-Vmax'
        return resp
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400


@app.get('/api/refined_data.json')
def refined_data():
    try:
        world = _get_pipeline()
        ci0 = request.args.get('ci0', -50, int)
        ci1 = request.args.get('ci1', 50, int)
        cj0 = request.args.get('cj0', -50, int)
        cj1 = request.args.get('cj1', 50, int)
        channels = {
            name: np.round(_refined_channel(world, ci0, ci1, cj0, cj1, i), 2).tolist()
            for i, name in enumerate(EDIT_CHANNELS)
        }
        return jsonify({'ci0': ci0, 'ci1': ci1, 'cj0': cj0, 'cj1': cj1, 'channels': channels})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.post('/api/coarse_tile')
def coarse_tile():
    try:
        data = request.get_json() or {}
        channel = int(data.get('channel', 0))
        ci = int(data['ci'])
        cj = int(data['cj'])
        value = float(data['value'])
        world = _get_pipeline()
        world.set_custom_conditioning_value(channel, ci, cj, _from_display_units(channel, value))
        return jsonify({'channel': channel, 'ci': ci, 'cj': cj, 'value': value})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.post('/api/coarse_brush')
def coarse_brush():
    try:
        data = request.get_json() or {}
        channel = int(data.get('channel', 0))
        world = _get_pipeline()
        stats = world.apply_conditioning_brush(
            channel=channel,
            center_ci=float(data['center_ci']),
            center_cj=float(data['center_cj']),
            radius=float(data.get('radius', 0)),
            strength=float(data.get('strength', 1.0)),
            mode=str(data.get('mode', 'set')),
            target_value=_from_display_units(channel, float(data.get('target_value', 0.0))),
            delta_value=_from_display_units(channel, float(data.get('delta_value', 100.0))),
            use_finalized_temp=bool(data.get('use_finalized_temp', False)),
        )
        if stats.get('min_value') is not None:
            stats['min_value'] = _to_display_units(channel, stats['min_value'])
            stats['max_value'] = _to_display_units(channel, stats['max_value'])
        return jsonify(stats)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400


@app.post('/api/rebuild')
def rebuild():
    try:
        world = _get_pipeline()
        _rebuild_pipeline(world)
        return jsonify({'ok': True, 'seed': str(world.seed)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400


@app.post('/api/reset_custom')
def reset_custom():
    try:
        world = _get_pipeline()
        channel = request.get_json(silent=True) or {}
        ch = channel.get('channel')
        world.clear_custom_conditioning(None if ch is None else int(ch))
        return jsonify({'ok': True})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400


@app.post('/api/cond_snr')
def cond_snr():
    try:
        world = _get_pipeline()
        data = request.get_json() or {}
        values = data.get('values')
        if not isinstance(values, list):
            return jsonify({'error': 'values must be a list'}), 400
        world.set_cond_snr(values)
        return jsonify({'cond_snr': list(world.kwargs['cond_snr'])})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 400


@app.post('/api/import_tiff')
def import_tiff():
    try:
        file = request.files.get('file')
        if file is None or not file.filename:
            return jsonify({'error': 'file required'}), 400
        if not file.filename.lower().endswith(('.tif', '.tiff')):
            return jsonify({'error': 'expected a .tif or .tiff file'}), 400

        channel = request.form.get('channel', 0, int)
        use_finalized_temp = bool(int(request.form.get('use_finalized_temp', 0)))
        values = _read_tiff_elevation(file)
        values = _from_display_units(channel, values)

        world = _get_pipeline()
        if use_finalized_temp and channel == 1:
            h, w = values.shape
            m = max(h, w)
            raw = world.get_conditioning_preview(0, m, 0, m)
            fin = world.get_conditioning_preview_finalized(0, m, 0, m)
            values = values + (raw[1][:h, :w] - fin[1][:h, :w])

        default_value = -1000.0 if channel == 0 else None
        world.set_custom_conditioning_import(channel, values, 0, 0, default_value=default_value)
        return jsonify({
            'channel': channel,
            'shape': [int(values.shape[0]), int(values.shape[1])],
            'origin': [0, 0],
            'default_value': default_value,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
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
@click.argument("model_path", default="xandergos/terrain-diffusion-30m")
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
@click.option("--port", type=int, default=int(os.getenv("PORT", "8081")))
@click.option("--kwarg", "extra_kwargs", multiple=True)
def main(model_path, hdf5_file, caching_strategy, cache_size, seed, device,
         batch_size, log_mode, torch_compile, dtype, host, port, extra_kwargs):
    """Terrain Editor web app"""
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
