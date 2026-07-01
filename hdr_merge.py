"""hdr_merge.py
HDR Merge (Gainmap) node for Comfy-FilmSimulator.

This file implements a lightweight, permissively-inspired HDR merge / gainmap routine.
It is an original implementation that is conceptually adapted from public projects
and literature (see README and comments). In particular this node was updated to
refer to the algorithms and ideas used by:
 - https://github.com/jb-jrdn/Hdr-Gainmap
 - https://github.com/zidage/AlcedoStudio

This code DOES NOT copy verbatim source from those projects. Instead it provides
a concise, ComfyUI-friendly node that computes a smooth gainmap from an HDR
and SDR pair and produces a merged HDR image plus a visualizable gainmap.

Author: repo maintainer (adapted for Comfy-FilmSimulator)
"""

import os
import numpy as np
import cv2
import torch

class HDRMergeNode:
    def __init__(self):
        self.name = "HDR Merge (Gainmap)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sdr_image": ("IMAGE",),
                "hdr_image": ("IMAGE",),
                "blur_radius": ("INT", {"default": 51, "min": 1, "max": 401, "step": 2}),
                "max_gain": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 64.0, "step": 0.1}),
                "mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("merged_hdr", "gainmap_vis", "info")
    FUNCTION = "merge_hdr"
    CATEGORY = "SmartHDR"

    def _to_hwc(self, img):
        # Accept torch tensors in either (C,H,W) or (H,W,C) with float types
        if isinstance(img, torch.Tensor):
            arr = img.cpu().numpy()
        else:
            arr = np.array(img)
        if arr.ndim == 3 and arr.shape[0] in (1,3) and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1,2,0))
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        return arr.astype(np.float32)

    def _to_tensor(self, arr):
        return torch.from_numpy(arr.astype(np.float32))

    def merge_hdr(self, sdr_image, hdr_image, blur_radius=51, max_gain=8.0, mix=1.0):
        """
        Merge an SDR image with an HDR image producing a smooth gainmap.
        - sdr_image, hdr_image: expected in linear (not gamma-corrected) float32 format in [0, +inf)
        - blur_radius: odd kernel size for Gaussian smoothing of the log-gain.
        - max_gain: clamp the gain to [1/max_gain, max_gain] to avoid extreme artifacts.
        - mix: how much of the computed gain to apply (0 = no change, 1 = full gain applied)

        Returns (merged_hdr_tensor, gainmap_visual_tensor, info_string)
        """
        # Safety and format conversions
        sdr = self._to_hwc(sdr_image)
        hdr = self._to_hwc(hdr_image)

        # Ensure same shape (if different, resize hdr to sdr)
        if hdr.shape[:2] != sdr.shape[:2]:
            hdr = cv2.resize(hdr, (sdr.shape[1], sdr.shape[0]), interpolation=cv2.INTER_LINEAR)

        eps = 1e-6
        # compute luminance for stability
        luma_sdr = 0.2126 * sdr[:,:,0] + 0.7152 * sdr[:,:,1] + 0.0722 * sdr[:,:,2]
        luma_hdr = 0.2126 * hdr[:,:,0] + 0.7152 * hdr[:,:,1] + 0.0722 * hdr[:,:,2]

        # Compute per-pixel ratio (gain) in log-domain to avoid extreme ratios
        # log_gain = log(hdr_l + eps) - log(sdr_l + eps)
        log_gain = np.log(np.maximum(luma_hdr, eps)) - np.log(np.maximum(luma_sdr, eps))

        # Smooth the log gain to create a spatially-coherent gainmap
        # ensure blur_radius is odd and at least 1
        k = int(max(1, blur_radius))
        if k % 2 == 0:
            k += 1
        try:
            smoothed_log_gain = cv2.GaussianBlur(log_gain, (k, k), 0)
        except Exception:
            # fallback to median blur if Gaussian fails for tiny kernels
            smoothed_log_gain = cv2.medianBlur((log_gain*1e6).astype(np.int32), 3).astype(np.float32) / 1e6

        gainmap = np.exp(smoothed_log_gain)

        # Clamp gainmap to avoid extreme amplification
        gainmap = np.clip(gainmap, 1.0 / max_gain, max_gain)

        # Apply the (optionally scaled) gainmap to SDR (per-channel)
        merged = sdr * (1.0 + (gainmap[...,None] - 1.0) * mix)

        # Keep merged non-negative
        merged = np.clip(merged, 0.0, None)

        # Prepare a visualizable gainmap (3-channel normalized to [0,1])
        gmin, gmax = float(gainmap.min()), float(gainmap.max())
        if gmax - gmin < 1e-6:
            gain_vis = np.ones_like(gainmap)
        else:
            gain_vis = (gainmap - gmin) / (gmax - gmin)
        gain_vis_rgb = np.dstack([gain_vis, gain_vis, gain_vis]).astype(np.float32)

        info = f"gain_range=({gmin:.3f},{gmax:.3f}), blur={k}, max_gain={max_gain}, mix={mix}"

        return (self._to_tensor(merged), self._to_tensor(gain_vis_rgb), info)
