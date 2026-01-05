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
        """
        try:
            # Simple heuristic mapping
            # High R/B ratio -> Low K (Warm)
            # Low R/B ratio -> High K (Cool)
            if b_gain == 0: return 0
            ratio = r_gain / b_gain
            k = 10000 / math.sqrt(ratio) * 0.6
            return int(k)
        except:
            return 0

    def read_dng(self, dng_path, linear_output, wb_mode, target_max_exposure):
        if not os.path.exists(dng_path):
            raise FileNotFoundError(f"DNG file not found: {dng_path}")
        
        try:
            with rawpy.imread(dng_path) as raw:
                # --- 1. Metadata Extraction (Safe Mode) ---
                # Fixed: Removed 'camera_make' which caused the crash.
                # raw.model usually contains the camera name.
                try:
                    camera_model = raw.model.decode('utf-8') if raw.model else "Unknown Camera"
                except:
                    camera_model = "Unknown Camera"

                # --- 2. White Balance Logic ---
                try:
                    camera_wb = list(raw.camera_whitebalance)
                    # Normalize to Green=1.0 for estimation
                    g_val = (camera_wb[1] + camera_wb[3]) / 2.0
                    if g_val > 0:
                        as_shot_r = camera_wb[0] / g_val
                        as_shot_b = camera_wb[2] / g_val
                        est_k = self.estimate_kelvin_from_multipliers(as_shot_r, as_shot_b)
                        wb_info_str = f"As Shot: R={as_shot_r:.2f} B={as_shot_b:.2f} (~{est_k}K)"
                    else:
                        wb_info_str = "As Shot: Unknown (Green=0)"
                except:
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
                    wb_info_str += " [IGNORED -> Using Auto]"
                elif wb_mode == "None (Raw Sensor)":
                    user_wb = [1.0, 1.0, 1.0, 1.0] 
                else:
                    # Presets
                    if wb_mode.startswith("Daylight"):
                        user_wb = list(raw.daylight_whitebalance) if hasattr(raw, 'daylight_whitebalance') and raw.daylight_whitebalance else [2.0, 1.0, 1.5, 1.0]
                    elif wb_mode.startswith("Tungsten"): user_wb = [1.5, 1.0, 2.5, 1.0]
                    elif wb_mode.startswith("Fluorescent"): user_wb = [1.8, 1.0, 2.2, 1.0]
                    elif wb_mode.startswith("Flash"): user_wb = [2.2, 1.0, 1.4, 1.0]

                # --- 3. Demosaicing (Decoding) ---
                white_level = raw.white_level
                if white_level == 0: white_level = 65535.0

                params = {
                    "gamma": (1.0, 1.0) if linear_output else (2.2, 4.5),
                    "no_auto_bright": True,
                    "output_bps": 16,
                    "bright": 1.0,
                    "user_sat": None
                }

                if use_camera_wb: params["use_camera_wb"] = True
                elif use_auto_wb: params["use_auto_wb"] = True
                elif user_wb:
                    params["use_camera_wb"] = False
                    params["use_auto_wb"] = False
                    params["user_wb"] = user_wb

                rgb_image = raw.postprocess(**params)
                
                # --- 4. Normalization (Float32) ---
                image_array = rgb_image.astype(np.float32)
                
                # Target Max Exposure: 0.5 leaves headroom for highlights
                image_array = image_array / white_level * target_max_exposure
                
                image_array = np.clip(image_array, 0.0, None)
                
                if len(image_array.shape) == 2:
                    image_array = np.stack([image_array] * 3, axis=-1)
                
                image_tensor = torch.from_numpy(image_array).unsqueeze(0)
                
                # Metadata String
                metadata = f"File: {os.path.basename(dng_path)}\n"
                metadata += f"Camera: {camera_model}\n"
                metadata += f"WB: {wb_info_str}\n"
                metadata += f"Exposure Gain: {target_max_exposure}x (Sensor Clip @ {target_max_exposure})"
                
                print(f"[SmartHDR] Loaded DNG: {camera_model}. Gain: {target_max_exposure}")
                
                return (image_tensor, metadata)
                
        except Exception as e:
            print(f"[DNG Reader Error] {e}")
            empty = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (empty, f"Error: {str(e)}")