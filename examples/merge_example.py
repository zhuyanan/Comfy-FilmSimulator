"""
examples/merge_example.py

Example usage of the FilmPipelineNode in this repo.

- Place this file in the repository under examples/
- Adjust `INPUT_DNG` (or supply an image tensor if you prefer)
- Run from the repo root with: python examples/merge_example.py
"""

import os
import sys

# If you want to import nodes as a package, you may need to add the repo root to sys.path.
# This example assumes you run it from the repository root so local imports work.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Import the pipeline node (local module import)
try:
    from film_pipeline import FilmPipelineNode
except Exception as e:
    print("Failed to import FilmPipelineNode from film_pipeline.py:", e)
    raise

# Replace this with a path to a DNG/RAW file you have available for testing.
INPUT_DNG = "/path/to/your/input.dng"


def main():
    pipeline = FilmPipelineNode()

    # Example: run the end-to-end pipeline
    hdr_out, sdr_preview, gainmap_vis, info = pipeline.run_pipeline(
        dng_path=INPUT_DNG,
        image=None,  # use dng_path; set to an IMAGE tensor if you want to pass one directly
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

    print("Pipeline info:")
    print(info)

    # hdr_out / sdr_preview / gainmap_vis are torch tensors (or None). If you want to inspect shape:
    try:
        import torch
        if isinstance(hdr_out, torch.Tensor):
            print("HDR output tensor shape:", hdr_out.shape)
        if isinstance(sdr_preview, torch.Tensor):
            print("SDR preview tensor shape:", sdr_preview.shape)
        if isinstance(gainmap_vis, torch.Tensor):
            print("Gainmap visual tensor shape:", gainmap_vis.shape)
    except Exception:
        pass


if __name__ == "__main__":
    main()
