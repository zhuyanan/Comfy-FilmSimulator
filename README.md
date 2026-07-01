---
## End-to-end pipeline: RAW -> Film -> Save SDR/HDR

This repository now includes a convenience pipeline node and helper nodes that make it easy to run the full workflow from a single node or chain the steps manually.

Nodes you can use

- `DNG Image Reader` — Load a DNG/RAW file (linear HDR output supported).
- `Film Simulation V4.1 (HDR Capable)` — Apply film simulation in a linear HDR workflow and generate both HDR frames and an SDR preview.
- `HDR Merge (Gainmap)` — Compute a smooth gainmap from an SDR/HDR pair and merge to produce a final HDR image suitable for AVIF/HEIC export.
- `Save SDR Preview (PNG/JPEG)` — Save the SDR preview (performs linear -> sRGB gamma if requested).
- `Save AVIF/HEIC HDR (Native)` — Save PQ-encoded AVIF/HEIC using Rec.2020 primaries and optional HDR metadata.
- `Film Pipeline (RAW -> Film -> Save)` — One-node orchestration that reads RAW, applies film simulation, optionally merges SDR+HDR, and saves SDR/HDR files.

Typical interactive workflows

- Manual (node graph):
  - `DNG Image Reader` -> `Film Simulation V4.1 (HDR Capable)` ->
    - connect `preview_sdr` -> `Save SDR Preview (PNG/JPEG)`
    - connect `hdr_image` -> `HDR Merge (Gainmap)` (optional) -> `Save AVIF/HEIC HDR (Native)`

- One-click (single node):
  - `Film Pipeline (RAW -> Film -> Save)` — set `dng_path` or supply an image tensor, configure film and merge parameters, and enable `save_sdr` / `save_hdr` as needed.

Usage example (programmatic)

Below is an example of how to call the FilmPipelineNode programmatically (this is a simplified outline; adjust paths and parameters to your setup):

```python
from comfy_film_simulator.film_pipeline import FilmPipelineNode

pipeline = FilmPipelineNode()

hdr_out, sdr_preview, gainmap_vis, info = pipeline.run_pipeline(
    dng_path="/path/to/input.dng",
    image=None,
    film_preset="Kodak Portra 400",
    wb_temperature_K=5600,
    wb_tint=0.0,
    auto_exposure=True,
    exposure=0.0,
    effect_strength=1.0,
    grain_power=1.0,
    halation_power=1.0,
    local_contrast=0.0,
    is_linear_input=True,
    run_merge=True,
    merge_blur_radius=51,
    merge_max_gain=8.0,
    merge_mix=1.0,
    merge_smoothing_mode="guided",
    save_sdr=True,
    sdr_filename_prefix="preview",
    sdr_format="PNG",
    sdr_quality=95,
    apply_srgb_gamma=True,
    save_hdr=True,
    hdr_filename_prefix="hdr",
    hdr_format="AVIF",
    hdr_quality=90,
    ref_white_nits=203.0
)

print(info)
```

Notes & recommendations

- Guided filter (edge-aware smoothing) is used by default for HDR merging when available (OpenCV contrib's ximgproc guidedFilter). If not available the pipeline falls back to Gaussian smoothing.
- The pipeline assumes linear scene-referred inputs for HDR processing. The SDR preview and SaveSDR node convert to sRGB for display when requested.
- For HDR export install `pillow-heif` and for guided filtering install `opencv-contrib-python` if you want edge-aware merging.

Attribution

The HDR merge and pipeline nodes were designed with conceptual inspiration from the following projects. This repository does not copy code verbatim from them; the implementations here are original and adapted for ComfyUI.

- https://github.com/jb-jrdn/Hdr-Gainmap
- https://github.com/zidage/AlcedoStudio

---
