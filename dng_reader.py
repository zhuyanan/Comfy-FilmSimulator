import rawpy
import numpy as np
import os
import torch
import math

class DNGImageReader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dng_path": ("STRING", {"default": "", "multiline": False}),
                "linear_output": ("BOOLEAN", {"default": True, "label_on": "Enabled (HDR/Linear)", "label_off": "Disabled (SDR/Gamma)"}),
                "wb_mode": (["As Shot", "Auto (Grey World)", "Daylight (5500K)", "Tungsten (2850K)", "Fluorescent (3800K)", "Flash", "None (Raw Sensor)"], {"default": "As Shot"}),
                "target_max_exposure": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0, "step": 0.05, "tooltip": "Physical Gain. 0.5 matches most camera previews. 1.0 is sensor clipping point."})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metadata")
    FUNCTION = "read_dng"
    CATEGORY = "SmartHDR"

    def estimate_kelvin_from_multipliers(self, r_gain, b_gain):
        """
        Estimate approximate Kelvin temperature from RGB multipliers.
        This is a very rough heuristic and only used for metadata display.
        """
        try:
            if b_gain == 0:
                return 0
            ratio = r_gain / b_gain
            k = 10000 / math.sqrt(ratio) * 0.6
            return int(k)
        except Exception:
            return 0

    def read_dng(self, dng_path, linear_output, wb_mode, target_max_exposure):
        # Validate path
        if not dng_path or not os.path.exists(dng_path):
            print(f"[DNG Reader Error] File not found: {dng_path}")
            empty = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (empty, f"Error: File not found: {dng_path}")
        
        try:
            with rawpy.imread(dng_path) as raw:
                # --- 1. White Balance Logic ---
                try:
                    camera_wb = list(raw.camera_whitebalance) if getattr(raw, 'camera_whitebalance', None) else [1.0, 1.0, 1.0, 1.0]
                    # safe average of green channels if present
                    if len(camera_wb) >= 4:
                        g_val = (camera_wb[1] + camera_wb[3]) / 2.0
                    elif len(camera_wb) >= 2:
                        g_val = camera_wb[1]
                    else:
                        g_val = 1.0

                    if g_val > 0:
                        as_shot_r = camera_wb[0] / g_val
                        as_shot_b = camera_wb[2] / g_val if len(camera_wb) >= 3 else 1.0
                        est_k = self.estimate_kelvin_from_multipliers(as_shot_r, as_shot_b)
                        wb_info_str = f"As Shot: R={as_shot_r:.2f} B={as_shot_b:.2f} (~{est_k}K)"
                    else:
                        wb_info_str = "As Shot: Unknown (Green=0)"
                except Exception:
                    wb_info_str = "As Shot: Unavailable"
                    camera_wb = [1, 1, 1, 1]

                user_wb = None
                use_camera_wb = False
                use_auto_wb = False

                if wb_mode == "As Shot":
                    use_camera_wb = True
                    wb_info_str += " [APPLIED]"
                elif wb_mode == "Auto (Grey World)":
                    use_auto_wb = True
                    wb_info_str += " [AUTO]"
                elif wb_mode == "None (Raw Sensor)":
                    user_wb = [1.0, 1.0, 1.0, 1.0]
                else:
                    wb_info_str = f"Preset: {wb_mode}"
                    if wb_mode.startswith("Daylight"):
                        user_wb = list(raw.daylight_whitebalance) if hasattr(raw, 'daylight_whitebalance') and raw.daylight_whitebalance else [2.0, 1.0, 1.5, 1.0]
                    elif wb_mode.startswith("Tungsten"):
                        user_wb = [1.5, 1.0, 2.5, 1.0]
                    elif wb_mode.startswith("Fluorescent"):
                        user_wb = [1.8, 1.0, 2.2, 1.0]
                    elif wb_mode.startswith("Flash"):
                        user_wb = [2.2, 1.0, 1.4, 1.0]

                # --- 2. Demosaicing (Decoding) ---
                white_level = float(getattr(raw, 'white_level', 65535.0) or 65535.0)
                if white_level <= 0:
                    white_level = 65535.0

                params = {
                    "gamma": (1.0, 1.0) if linear_output else (2.2, 4.5),
                    "no_auto_bright": True,
                    "output_bps": 16,
                    "bright": 1.0,
                    "user_sat": None
                }

                if use_camera_wb:
                    params["use_camera_wb"] = True
                elif use_auto_wb:
                    params["use_auto_wb"] = True
                elif user_wb:
                    params["use_camera_wb"] = False
                    params["use_auto_wb"] = False
                    params["user_wb"] = user_wb

                # raw.postprocess can raise; keep in try/except
                rgb_image = raw.postprocess(**params)

                # --- 3. Normalization (Float32) ---
                image_array = rgb_image.astype(np.float32)
                # avoid division by zero
                white_level_safe = max(white_level, float(np.max(image_array)) if image_array.size else 65535.0)
                image_array = image_array / white_level_safe * float(target_max_exposure)
                image_array = np.clip(image_array, 0.0, None)

                # ensure shape HxWx3
                if image_array.ndim == 2:
                    image_array = np.stack([image_array] * 3, axis=-1)
                elif image_array.ndim == 3 and image_array.shape[2] != 3 and image_array.shape[0] in (1, 3):
                    # possibly channels-first (C,H,W)
                    image_array = np.transpose(image_array, (1, 2, 0))

                # ensure float32 contiguous
                image_array = np.ascontiguousarray(image_array.astype(np.float32))
                image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()

                # --- 4. Metadata Extraction ---
                metadata = self.extract_metadata(raw, dng_path, wb_info_str, target_max_exposure, linear_output)

                print(f"[SmartHDR] Loaded DNG: {os.path.basename(dng_path)}. Gain: {target_max_exposure}")

                return (image_tensor, metadata)

        except Exception as e:
            print(f"[DNG Reader Error] {e}")
            empty = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (empty, f"Error: {str(e)}")

    def extract_metadata(self, raw, dng_path, wb_info_str, target_max_exposure, linear_output):
        try:
            sizes = getattr(raw, 'sizes', None)
            if sizes is not None:
                width = getattr(sizes, 'width', '?')
                height = getattr(sizes, 'height', '?')
            else:
                width = getattr(raw, 'raw_image_visible', None)
                if width is not None:
                    # best-effort
                    height = raw.raw_image_visible.shape[0]
                    width = raw.raw_image_visible.shape[1]
                else:
                    width = '?'
                    height = '?'

            filters = getattr(sizes, 'filters', None) if sizes is not None else None
            if filters is not None:
                if filters == 0:
                    bayer_pattern_str = "No filter (Monochrome)"
                elif filters == 9:
                    bayer_pattern_str = "RGGB"
                elif filters == 273:
                    bayer_pattern_str = "GRBG"
                elif filters == 1536:
                    bayer_pattern_str = "BGGR"
                elif filters == 6144:
                    bayer_pattern_str = "GBRG"
                else:
                    bayer_pattern_str = f"Unknown ({filters})"
            else:
                bayer_pattern_str = "Unknown"

            try:
                color_desc = raw.color_desc.decode('utf-8') if hasattr(raw.color_desc, 'decode') else str(raw.color_desc)
            except Exception:
                color_desc = str(getattr(raw, 'color_desc', 'Unknown'))

            try:
                camera_model = raw.model.decode('utf-8') if hasattr(raw, 'model') and raw.model else 'Unknown'
            except Exception:
                camera_model = 'Unknown'

            try:
                camera_make = raw.camera_make.decode('utf-8') if hasattr(raw, 'camera_make') and raw.camera_make else 'Unknown'
            except Exception:
                camera_make = 'Unknown'

            metadata_str = f"File: {os.path.basename(dng_path)}\nWidth: {width}\nHeight: {height}\nCamera: {camera_model}\nCamera Maker: {camera_make}\nBayer Pattern: {bayer_pattern_str}\nColor Description: {color_desc}\nWB: {wb_info_str}\nExposure Gain: {target_max_exposure}x\nLinear Output Mode: {'Enabled' if linear_output else 'Disabled'}"

            try:
                black_level = getattr(raw, 'black_level', 'Unknown')
                metadata_str += f"\nBlack Level: {black_level}"
                white_level = getattr(raw, 'white_level', 'Unknown')
                metadata_str += f"\nWhite Level: {white_level}"
            except Exception:
                pass

            return metadata_str
        except Exception as e:
            return f"Error extracting metadata: {str(e)}"
