# Terrain Editor

A web UI to paint or import terrain and climate data, then regenerate the high-resolution world from your edits.

## Run

From the repo root:

```bash
python -m terrain_diffusion edit
```

Optional: `--port 8080`, `--no-compile` (faster startup, slower inference). Default model is `xandergos/terrain-diffusion-30m`.

---

## Two panels

| Left | Right |
|------|--------|
| **Coarse map** (conditioning) | **Detail map** (rebuilt output) |
| What you edit: elevation, temperature, precipitation, etc. | What the model generates from that guidance |
| Updates as soon as you brush or import | Updates only when you click **Refresh Detail (Rebuild)** |

You can switch the left panel to **Refined** to see the coarse model’s output before the detail stage, to see how much influence your guidance is having. The right panel can show relief, elevation, or climate layers.

---

## Channels

You edit one of five conditioning channels at a time:

- **Elev** — Elevation (m).
- **Temp** — Mean temperature (°C).
- **T std** — Temperature variability (°C).
- **Precip** — Annual precipitation (mm).
- **P CV** — Precipitation variability (%).

Select a channel with the buttons; brush and import apply to the selected channel only.

---

## Tools

- **Select** — Click the coarse map to choose which detail tile is shown on the right. Click the detail map to pan.
- **Brush** — Paint on the coarse map. Modes:
  - **Set** — Blend toward a target value.
  - **Raise** / **Lower** — Add or subtract a step per stroke.
  - **Smooth** — Local smoothing.
  - **Noise** — Add random variation.

**Brush Size** and **Brush Strength** control radius and how much each stroke changes the map. **Target Value** is used by Set; **Raise/Lower Step** is used by Raise and Lower.

---

## Refinement strength

The sliders (**Refinement Strength**) control how closely the model follows your conditioning. Higher values = stronger adherence. Changes apply only after you click **Refresh Detail (Rebuild)**.

---

## Import TIFF

**Import TIFF** loads a single-band GeoTIFF (e.g. from [azgaar-to-tiff](https://github.com/xandergos/terrain-diffusion/blob/main/terrain_diffusion/inference/utils/azgaar_to_tiff.py)) into the current channel. The image is placed at coarse origin (0, 0). For elevation, areas outside the TIFF are set to -1000 m; for climate channels, the base synthetic map is used outside the import. **Reset Channel** clears imports and brush edits for the current channel (or all channels).

---

## Refresh Detail (Rebuild)

Click **Refresh Detail (Rebuild)** to regenerate the world from the current conditioning map and refinement settings. Until you do, the detail panel does not reflect brush or import changes. Rebuild can take a while; the pipeline runs the coarse model, then the latent and decoder stages.

---

## Edits are guidance

The model uses your conditioning as **guidance**, not strict constraints. The coarse model can soften or partly ignore edits. If the detail view doesn’t match the left panel:

- Click **Refresh Detail (Rebuild)** if you haven’t.
- Increase **Refinement Strength** for the channel you edited.
- Ensure the right-panel view (e.g. Relief vs Temperature) shows the channel you changed.

---

## Main files

- `terrain_diffusion/inference/editor/server.py` — Flask backend
- `terrain_diffusion/inference/editor/static/index.html` — Frontend

The editor uses the same `WorldPipeline` as the explorer and APIs; see `world_pipeline.py` for the generation stack.
