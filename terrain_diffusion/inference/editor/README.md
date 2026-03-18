# Terrain Editor

This editor is a web UI for modifying the coarse-model conditioning map and then rebuilding the world from those staged edits.

## Run

From the repo root:

```bash
python3.11 -m terrain_diffusion edit --port 8080 --no-compile
```

The editor uses the same `WorldPipeline` as the rest of inference.

## Mental Model

There are two different things shown in the editor:

1. The left map is the **conditioning preview**.
2. The right map is the **rebuilt detail output** from the current world.

Edits are applied to the conditioning preview immediately, but the detail map does **not** change until you click `Refresh Detail (Rebuild)`.

## What Is Editable

The editor currently lets you modify the 5 conditioning channels that feed the coarse model:

- `Elev`
- `Temp`
- `T std`
- `Precip`
- `P CV`

These are the channels returned by `WorldPipeline.get_conditioning_preview()`.

The editor does **not** directly edit `p5`.

## How Rebuild Works

On rebuild, the pipeline:

1. Samples the base synthetic conditioning map.
2. Applies staged imports and brush edits to the selected conditioning channels.
3. Converts that edited conditioning map into the coarse-model input.
4. Runs the coarse model.
5. Rebuilds the latent stage.
6. Rebuilds the decoder stage.

So the coarse model is still used in the background. The editor does not bypass it.

## Important Caveat

Edits are currently **guidance**, not hard constraints.

That means:

- changing the conditioning map influences the rebuilt coarse world
- but the coarse model can still partially ignore or soften those edits
- some rebuilt changes may therefore look weaker than the staged preview

This is especially noticeable when:

- `cond_snr` values are low
- the edited region is small
- the selected detail visualization does not directly show the edited channel

## `p5`

`p5` is still rebuilt because it is one of the coarse model outputs.

However:

- `p5` is not shown as an editable conditioning channel
- `p5` changes only indirectly through the coarse-model response to the edited conditioning inputs

## `cond_snr`

The editor exposes sliders for `cond_snr`.

These values are staged into `world.kwargs['cond_snr']` and only take effect after `Refresh Detail (Rebuild)`, because the coarse stage must be rebuilt with the new conditioning strength.

Higher values generally make the coarse model follow the conditioning map more strongly.

## Brush Tools

The brush operates on the currently selected conditioning channel.

Modes:

- `Set`: blend toward a target value
- `Raise`: add a positive delta
- `Lower`: subtract a delta
- `Smooth`: local blur-like smoothing

Controls:

- `Brush Size`: radius in coarse tiles
- `Brush Strength`: per-stroke blending strength
- `Target`: used by `Set`
- `Step`: used by `Raise` and `Lower`

Dragging paints repeatedly across coarse tiles.

## TIFF Import

Import applies to the currently selected channel.

Behavior:

- the TIFF is centered on the current coarse view
- for `Elev`, values outside the imported bounds default to `-1000 m`
- for climate channels, outside the imported bounds the editor falls back to the base synthetic conditioning map

Only the first TIFF band is used.

## Refresh Button

`Refresh Detail (Rebuild)` is the authoritative update step.

It calls the rebuild endpoint, which reconstructs the pipeline hierarchy and then reloads the detail tile. If something changes in the preview but not in the detail view, the most common reasons are:

- rebuild was not clicked
- the selected detail mode does not make that channel obvious
- the coarse model did not follow the conditioning strongly enough

## Main Files

- `terrain_diffusion/inference/editor/server.py`
- `terrain_diffusion/inference/editor/static/index.html`
- `terrain_diffusion/inference/world_pipeline.py`

