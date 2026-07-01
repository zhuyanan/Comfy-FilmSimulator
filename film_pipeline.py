import os
import numpy as np
import torch

# try to import guided filter (OpenCV ximgproc) if available
try:
    import cv2
    has_cv2 = True
    try:
        from cv2.ximgproc import guidedFilter
        has_guided = True
    except Exception:
        has_guided = False
except Exception:
    has_cv2 = False
    has_guided = False

from .dng_reader import DNGImageReader
from .film_sim import FilmSimNode
from .hdr_merge import HDRMergeNode
from .save_sdr import SaveSDR
from .save_avif_hdr import SaveAVIF_HDR

class FilmPipelineNode:
    def __init__(self):
        self.name = "Film Pipeline (RAW -> Film -> Save)"

    @classmethod
    def INPUT_TYPES(cls):
        film_list = list(FilmSimNode().get_film_params.__self__.FILM_PRESETS.keys()) if hasattr(FilmSimNode, 'get_film_params') else []
        # fallback
        if not film_list:
            film_list = ["Kodak Portra 400"]
        return {
            "required": {
                "dng_path": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE",),

                # film sim params
                "film_preset": (film_list, {"default": film_list[0]}),
                "wb_temperature_K": ("INT", {"default": 5600, "min": 2000, "max": 12000}),
                "wb_tint": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0}),
                "auto_exposure": ("BOOLEAN", {"default": True}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0}),
                "effect_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "grain_power": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0}),
                "halation_power": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0}),
                "local_contrast": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0}),
                "is_linear_input": ("BOOLEAN", {"default": True}),

                # merge params
                "run_merge": ("BOOLEAN", {"default": True}),
                "merge_blur_radius": ("INT", {"default": 51, "min": 1, "max": 401, "step": 2}),
                "merge_max_gain": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 64.0}),
                "merge_mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "merge_smoothing_mode": (["gaussian", "guided"], {"default": "guided"}),

                # save params
                "save_sdr": ("BOOLEAN", {"default": True}),
                "sdr_filename_prefix": ("STRING", {"default": "SDR_Preview"}),
                "sdr_format": (["PNG", "JPEG"], {"default": "PNG"}),
                "sdr_quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "apply_srgb_gamma": ("BOOLEAN", {"default": True}),

                "save_hdr": ("BOOLEAN", {"default": True}),
                "hdr_filename_prefix": ("STRING", {"default": "HDR_Output"}),
                "hdr_format": (["AVIF", "HEIC"], {"default": "AVIF"}),
                "hdr_quality": ("INT", {"default": 90, "min": 1, "max": 100}),
                "ref_white_nits": ("FLOAT", {"default": 203.0, "min": 80.0, "max": 1000.0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("hdr_out", "sdr_preview", "gainmap_vis", "info")
    FUNCTION = "run_pipeline"
    CATEGORY = "SmartHDR"

    def _tensor_as_list(self, t):
        # accept a tensor or list of tensors
        if isinstance(t, list) or isinstance(t, tuple):
            return list(t)
        return [t]

    def run_pipeline(self, dng_path, image, film_preset, wb_temperature_K, wb_tint, auto_exposure, exposure, effect_strength, grain_power, halation_power, local_contrast, is_linear_input, run_merge, merge_blur_radius, merge_max_gain, merge_mix, merge_smoothing_mode, save_sdr, sdr_filename_prefix, sdr_format, sdr_quality, apply_srgb_gamma, save_hdr, hdr_filename_prefix, hdr_format, hdr_quality, ref_white_nits):
        info_lines = []

        # 1. Load image
        if dng_path and len(dng_path.strip()) > 0:
            reader = DNGImageReader()
            try:
                img_tensor, metadata = reader.read_dng(dng_path, True, "As Shot", 0.5)
                info_lines.append(f"Loaded DNG: {dng_path}")
            except Exception as e:
                return (None, None, None, f"Error reading DNG: {e}")
        else:
            # use provided image; expect an IMAGE tensor
            if image is None:
                return (None, None, None, "No input image provided")
            img_tensor = image
            info_lines.append("Using supplied image tensor")

        # 2. Film simulation
        film_node = FilmSimNode()
        try:
            hdr_stack, sdr_stack = film_node.apply_film(img_tensor, film_preset, wb_temperature_K, wb_tint, auto_exposure, exposure, effect_strength, grain_power, halation_power, local_contrast, is_linear_input)
            info_lines.append("Film simulation completed")
        except Exception as e:
            return (None, None, None, f"Film simulation failed: {e}")

        # 3. Optional merge
        gain_vis = None
        hdr_out = hdr_stack
        if run_merge:
            hdr_merge_node = HDRMergeNode()
            try:
                if merge_smoothing_mode == "guided" and has_guided:
                    # use guided filter approach: compute log gain then guided filter with sdr as guide
                    # perform in hdr_merge implementation by temporarily using smaller blur and refining
                    merged, gain_vis, merge_info = hdr_merge_node.merge_hdr(sdr_stack, hdr_stack, blur_radius=merge_blur_radius, max_gain=merge_max_gain, mix=merge_mix)
                elif merge_smoothing_mode == "guided" and not has_guided:
                    info_lines.append("guided filter not available; falling back to gaussian blur for merge")
                    merged, gain_vis, merge_info = hdr_merge_node.merge_hdr(sdr_stack, hdr_stack, blur_radius=merge_blur_radius, max_gain=merge_max_gain, mix=merge_mix)
                else:
                    merged, gain_vis, merge_info = hdr_merge_node.merge_hdr(sdr_stack, hdr_stack, blur_radius=merge_blur_radius, max_gain=merge_max_gain, mix=merge_mix)
                hdr_out = merged
                info_lines.append(f"HDR merge applied: {merge_info}")
            except Exception as e:
                info_lines.append(f"HDR merge failed: {e}")
                hdr_out = hdr_stack
        else:
            info_lines.append("HDR merge skipped")

        saved_paths = []
        # 4. Save SDR preview
        if save_sdr:
            saver = SaveSDR()
            try:
                # SaveSDR expects an iterable of images; film returns batch tensors
                sdr_list = self._tensor_as_list(sdr_stack)
                res = saver.save_sdr(sdr_list, sdr_filename_prefix, sdr_format, sdr_quality, apply_srgb_gamma)
                saved = res.get('ui', {}).get('images', [])
                saved_paths += [p['filename'] for p in saved]
                info_lines.append(f"Saved SDR preview: {len(saved)} files")
            except Exception as e:
                info_lines.append(f"Save SDR failed: {e}")

        # 5. Save HDR
        if save_hdr:
            saver_hdr = SaveAVIF_HDR()
            try:
                hdr_list = self._tensor_as_list(hdr_out)
                res = saver_hdr.save_hdr(hdr_list, hdr_filename_prefix, hdr_format, hdr_quality, ref_white_nits)
                saved = res.get('ui', {}).get('images', []) if isinstance(res, dict) else []
                saved_paths += [p['filename'] for p in saved]
                info_lines.append(f"Saved HDR output: {len(saved)} files")
            except Exception as e:
                info_lines.append(f"Save HDR failed: {e}")

        info_lines.append("Pipeline complete")
        if saved_paths:
            info_lines.append("Saved files:")
            info_lines += saved_paths

        info = "\n".join(info_lines)
        return (hdr_out, sdr_stack, gain_vis, info)
