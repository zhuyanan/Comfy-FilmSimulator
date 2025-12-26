# Comfy-FilmSimulator

ComfyUI node: realistic, adaptive film simulation for photographic and cinematic looks.

This repository provides a ComfyUI-compatible node implementation that simulates film stock response, grain, bloom/halation and tone mapping. It includes an extendable preset library (`films.json`) and a PyTorch/OpenCV-based processing node (`__init__.py`) that exposes controls for film type, tone mapping, exposure, grain and halation.

---

## Quick highlights

- Node name (displayed in ComfyUI): **Film Simulation V3 (Adaptive)**
- Node category: `FilmSim/Film`
- Key files:
  - `__init__.py` — the ComfyUI node implementation (expects ComfyUI IMAGE tensors, uses PyTorch + OpenCV)
  - `films.json` — presets database of film stocks (color & B/W) with tone curve, grain and optical params
  - `LICENSE` — project license
- Tone-mapping options: `filmic`, `reinhard`, `none`
- Presets include many classic stocks (e.g. Kodak Portra 160/400/800, Fuji Pro 400H, Tri-X 400, T-Max, LomoChrome, etc.)

---

## Installation (ComfyUI)

1. Install the required Python packages if not already present (run in the same environment ComfyUI uses):

```bash
pip install numpy opencv-python torch
```

- Note: Install `torch` according to your platform/GPU configuration (see https://pytorch.org).
- If your ComfyUI environment already has these, skip re-installing them.

2. Install the node:
   - Copy `__init__.py` and `films.json` into a folder under your ComfyUI `custom_nodes` directory. Example:

```text
<ComfyUI root>/
  custom_nodes/
    FilmSim/
      __init__.py
      films.json
```

3. Restart ComfyUI. The node will appear under category `FilmSim/Film` named `Film Simulation V3 (Adaptive)`.

---

## Inputs & Parameters (as exposed in the node)

- image (IMAGE) — input image tensor(s) (ComfyUI IMAGE)
- film_type (select) — choose from presets listed in `films.json` (default: `NC200 (Fuji C200 style)` fallback)
- tone_mapping (select) — `filmic` (default), `reinhard`, `none`
- exposure (FLOAT) — manual EV compensation, range roughly -3.0 .. +3.0 (adds to preset `exposure_bias`)
- grain_factor (FLOAT) — global multiplier for grain strength (default 1.0)
- halation_factor (FLOAT) — controls bloom/halation strength (default 1.0)

Return: processed IMAGE tensor.

---

## How it works (high level)

- Input normalization & channel split (supports 3/4-channel inputs).
- Per-preset color matrix converts incoming channels to film "lux" channels.
- Adaptive filmic curve:
  - Anchors mid-gray to avoid brightness drift with gamma changes.
  - Uses preset tone curve (A-F, gamma, exposure_bias) with an adaptive whitepoint scaling.
  - Outputs a matte look clamped to [0.02, 0.98] to emulate film/paper limits.
- Grain generator:
  - Per-channel randomized noise with weighting that depends on local luminance.
  - Cross-talk mixes grain between channels to simulate analog film interactions.
- Optical effects:
  - Bloom/halation applied with configurable radii derived from image size and sensitivity factor.
  - Simple vignette/optical weighting parameters present in presets.
- Presets:
  - `films.json` contains many stocks with `type`, `matrix`, `sens_factor`, `grain_base`, `opt_*` and `curve` values.

---

## Example usage (within ComfyUI)

1. Add an image source node.
2. Add the "Film Simulation V3 (Adaptive)" node (category: FilmSim/Film).
3. Connect the image output to the `image` input on the FilmSim node.
4. Select a `film_type` (e.g. "Kodak Portra 400"), choose `filmic` or `reinhard` tone mapping.
5. Tweak `exposure`, `grain_factor` and `halation_factor` to taste.
6. Connect the node output to your compositor or final render node.

---

## Preset editing & adding new stocks

- Presets are JSON objects in `films.json`.
- Each preset includes:
  - `type`: `"color"` or `"bw"`
  - `matrix`: color transform coefficients (9 values for RGB mapping or last three entries used for B/W)
  - `sens_factor`: affects bloom radius and halo strength
  - `grain_base`: base grain strength for R/G/B/L channels and cross-talk factor
  - `opt_r/opt_g/opt_b` or `opt_l`: per-channel optical composition coefficients
  - `curve`: tone mapping parameters {A, B, C, D, E, F, gamma, exposure_bias}
- To create a new preset: copy an existing block, adjust parameters, save, and restart ComfyUI.

---

## Development notes

- The node uses OpenCV (cv2) for filtering and NumPy for array ops; PyTorch tensors are used for I/O with ComfyUI.
- The code tries to fall back to an embedded default preset if `films.json` is missing or fails to load.
- Timestamps (tick count) seed the grain generator for per-frame variation.

Suggested dev setup:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if you add one
# or at minimum:
pip install numpy opencv-python torch
```

There are no automated tests in the repository currently; visual verification (example input -> node -> compare output) is the primary validation method.

---

## License

This repository contains a `LICENSE` file — please refer to it for terms. (The repo includes a LICENSE in the root.)

---

## Attribution & Contact

Maintainer: zhuyanan  
Repo: https://github.com/zhuyanan/Comfy-FilmSimulator

