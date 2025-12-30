import torch
import numpy as np
import json
import os
import sys

# --- Check Dependencies ---
try:
    import cv2
except ImportError:
    print("\n[FilmSimNode Error] 'opencv-python' is missing!")
    print("Please install it using: pip install opencv-python\n")
    raise

# --- Load Presets ---
def load_presets():
    try:
        current_dir = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        current_dir = os.getcwd()
        
    json_path = os.path.join(current_dir, "films.json")
    
    # Minimal Fallback
    default_presets = {
        "Default Color": {
            "type": "color",
            "matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, 0, 0],
            "sens_factor": 1.0,
            "grain_base": [0.10, 0.10, 0.10, 0.05],
            "opt_r": [1.0, 1.0, 1.0],
            "opt_g": [1.0, 1.0, 1.0],
            "opt_b": [1.0, 1.0, 1.0],
            "curve": {"A": 0.15, "B": 0.50, "C": 0.10, "D": 0.20, "E": 0.02, "F": 0.30, "gamma": 2.0, "exposure_bias": 0.0}
        }
    }

    if not os.path.exists(json_path):
        return default_presets

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"[FilmSimNode] Error loading films.json: {e}")
        return default_presets

FILM_PRESETS = load_presets()

class FilmSimNode:
    """
    Film Simulation V4.0 (Final)
    - Coherent Grain with Crosstalk
    - Local Contrast Enhancement (Clarity)
    - High-Fidelity Tone Mapping with Highlight Protection
    - Responsive Split Toning
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        film_list = list(FILM_PRESETS.keys())
        default_film = film_list[0] if film_list else "Default Color"

        return {
            "required": {
                "image": ("IMAGE",),
                "film_type": (film_list, {"default": default_film}),
                "tone_mapping": (["filmic", "reinhard", "none"], {"default": "filmic"}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1, "tooltip": "EV Offset"}),
                "grain_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "halation_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_film"
    CATEGORY = "FilmSim/Film"

    def get_film_params(self, film_type_name):
        # Create a shallow copy to prevent modifying global state
        p = FILM_PRESETS.get(film_type_name)
        if p is None:
             p = list(FILM_PRESETS.values())[0]
        p = p.copy() 
        
        # Ensure matrix has enough elements
        if "matrix" not in p: p["matrix"] = [1.0,0,0, 0,1.0,0, 0,0,1.0, 0,0,0]
        while len(p["matrix"]) < 12: p["matrix"].append(0.0)

        # Ensure grain_base has 4 elements (crosstalk)
        if "grain_base" not in p: p["grain_base"] = [0.1, 0.1, 0.1, 0.0]
        while len(p["grain_base"]) < 4: p["grain_base"].append(0.0)

        # Default options
        if p["type"] == "color":
             if "opt_r" not in p: p["opt_r"] = [1.0, 1.0, 1.0]
             if "opt_g" not in p: p["opt_g"] = [1.0, 1.0, 1.0]
             if "opt_b" not in p: p["opt_b"] = [1.0, 1.0, 1.0]
        elif p["type"] == "bw":
             if "opt_l" not in p: p["opt_l"] = [1.0, 1.0, 1.0]
             
        if "curve" in p and "exposure_bias" not in p["curve"]:
            p["curve"]["exposure_bias"] = 0.0
            
        return p

    def luminance_calc(self, image, p):
        if len(image.shape) == 2:
            c1 = c2 = c3 = image
        elif image.shape[2] == 4:
            c1, c2, c3, _ = cv2.split(image)
        elif image.shape[2] == 3:
            c1, c2, c3 = cv2.split(image)
        else:
            c1 = c2 = c3 = image[:,:,0]

        m = p["matrix"]
        if p["type"] == "color":
            lux_r = m[0]*c1 + m[1]*c2 + m[2]*c3
            lux_g = m[3]*c1 + m[4]*c2 + m[5]*c3
            lux_b = m[6]*c1 + m[7]*c2 + m[8]*c3
            return lux_r, lux_g, lux_b, 0
        else:
            lux_total = m[9]*c1 + m[10]*c2 + m[11]*c3
            return None, None, None, lux_total

    def generate_grain(self, channel, grain_intensity, seed_shift=0):
        if channel is None: return 0
        np.random.seed(int(self.time_hash + seed_shift) % 2**32)
        
        noise = np.random.normal(0, 1, channel.shape).astype(np.float32)
        
        # Texture visibility weights based on luminance
        weights = (0.5 - np.abs(channel - 0.5)) * 2
        weights = np.clip(weights, 0.05, 0.9)
        
        weighted_noise = noise * weights * grain_intensity
        # Soft blur to simulate silver halide size
        weighted_noise = cv2.GaussianBlur(weighted_noise, (3, 3), 1)
        return np.clip(weighted_noise, -1, 1)

    def apply_local_contrast(self, image, strength=0.2):
        # Unsharp Masking Logic for Clarity
        blur_radius = int(min(image.shape[0], image.shape[1]) * 0.01) | 1
        blurred = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
        details = image - blurred
        return image + details * strength

    def adjust_saturation(self, img, saturation):
        # Fast Saturation: Color = Luma + (Color - Luma) * Sat
        if saturation == 1.0: return img
        
        luma = img[:,:,0] * 0.2126 + img[:,:,1] * 0.7152 + img[:,:,2] * 0.0722
        luma = np.stack([luma, luma, luma], axis=2)
        
        return luma + (img - luma) * saturation

    def filmic_curve(self, x, c, manual_exposure=0.0):
        if x is None: return None
        x = np.maximum(x, 0)
        
        # 1. Adaptive Gain (Hable anchor)
        base_scaling = 10.0 * (0.18 ** 2.0)
        current_mid_response = 0.18 ** c["gamma"]
        standard_adaptive_scale = base_scaling / (current_mid_response + 1e-6)
        
        json_bias = c.get("exposure_bias", 0.0)
        total_ev = json_bias + manual_exposure
        exposure_mult = 2.0 ** total_ev
        
        curr_x = (standard_adaptive_scale * exposure_mult) * (x ** c["gamma"]) 
        
        A, B, C, D, E, F = c["A"], c["B"], c["C"], c["D"], c["E"], c["F"]
        
        def curve_fn(t):
            denominator = (t * (A * t + B) + D * F) + 1e-6 
            numerator = (t * (A * t + C * B) + D * E)
            return (numerator / denominator) - (E / F)

        mapped = curve_fn(curr_x)
        
        # White point normalization
        W_linear = (1.0 / 0.18) ** c["gamma"]
        W_input = W_linear * standard_adaptive_scale
        white_scale = curve_fn(np.array([W_input]))
        
        # [Fixed] Ensure scalar for boolean checks
        white_scale_raw = curve_fn(np.array([W_input]))
        if isinstance(white_scale_raw, np.ndarray):
             white_scale = float(white_scale_raw.item()) if white_scale_raw.size == 1 else 1.0
        else:
             white_scale = float(white_scale_raw)

        if white_scale < 1e-4: white_scale = 1.0 
            
        normalized = mapped / white_scale
        
        # Standard Matte Look (Fixed range, tints handle the rest)
        target_black = 0.01
        target_white = 0.99
        return normalized * (target_white - target_black) + target_black

    def reinhard_curve(self, x, gamma):
        if x is None: return None
        mapped = x * (x / (1.0 + x))
        mapped = np.power(mapped, 1.0/gamma)
        return np.clip(mapped, 0, 1)

    def apply_split_toning(self, r, g, b, tint_data):
        if not tint_data:
            return r, g, b
            
        s_rgb = tint_data.get("shadows", [0, 0, 0])
        h_rgb = tint_data.get("highlights", [0, 0, 0])
        power = tint_data.get("range", 2.0)

        if all(v == 0 for v in s_rgb) and all(v == 0 for v in h_rgb):
            return r, g, b

        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        luma = np.clip(luma, 0, 1)

        s_mask = np.power(1.0 - luma, power)
        h_mask = np.power(luma, power)

        r_new = r + (s_rgb[0] * s_mask) + (h_rgb[0] * h_mask)
        g_new = g + (s_rgb[1] * s_mask) + (h_rgb[1] * h_mask)
        b_new = b + (s_rgb[2] * s_mask) + (h_rgb[2] * h_mask)

        return np.clip(r_new, 0, 1), np.clip(g_new, 0, 1), np.clip(b_new, 0, 1)

    def process_film(self, image, film_type, tone_mapping, exposure, grain_factor, halation_factor):
        self.time_hash = int(cv2.getTickCount())
        output_images = []
        
        for i in range(image.shape[0]):
            img_np = image[i].cpu().numpy()
            if img_np.dtype != np.float32 and img_np.dtype != np.float64:
                 img_np = img_np.astype(np.float32)

            height, width = img_np.shape[:2]
            p = self.get_film_params(film_type)
            
            # --- 1. Physics: Luminance & Sensitivity ---
            lux_r, lux_g, lux_b, lux_total = self.luminance_calc(img_np, p)
            
            if p["type"] == "color":
                avg_lux_map = (lux_r + lux_g + lux_b) / 3.0
            else:
                avg_lux_map = lux_total

            # Sensitivity Map: Dark areas are more sensitive to grain/bloom visibility
            sens_map = (1.0 - avg_lux_map) * 0.75 + 0.10
            sens_map = np.clip(sens_map, 0.10, 0.7)
            # Scalar for Gaussian kernels
            sens_scalar = float(np.mean(sens_map))
            
            # --- 2. Physics: Halation / Bloom ---
            scale_ratio = max(height, width) / 3000.0
            rads = int(20 * (sens_scalar**2) * p["sens_factor"] * 2.0 * scale_ratio) 
            rads = max(1, rads)
            ksize = rads * 2 + 1
            
            base_diffusion = 0.05 * p["sens_factor"]
            halo_str = 23 * sens_scalar**2 * p["sens_factor"] * halation_factor

            def apply_bloom(lux_channel, sigma_mult):
                if lux_channel is None: return None
                weights = (base_diffusion + lux_channel**2) * sens_map
                weights = np.clip(weights, 0, 1)
                k = int(ksize * 3) | 1
                sigma = sens_scalar * sigma_mult * scale_ratio
                bloom = cv2.GaussianBlur(lux_channel * weights, (k, k), sigma)
                effect = bloom * weights * halo_str
                return effect / (1.0 + effect)

            # --- 3. Physics: Grain (Coherent + Crosstalk) ---
            g_str = [g * grain_factor for g in p["grain_base"]]
            
            gn_r, gn_g, gn_b = 0, 0, 0
            if grain_factor > 0:
                if p["type"] == "color":
                    # Master texture (Luminance based)
                    gn_master = self.generate_grain(avg_lux_map, 1.0, seed_shift=0)
                    
                    # Small decoherence for RGB
                    color_decoherence = 0.2
                    gn_var_r = self.generate_grain(lux_r, 1.0, seed_shift=100)
                    gn_var_g = self.generate_grain(lux_g, 1.0, seed_shift=200)
                    gn_var_b = self.generate_grain(lux_b, 1.0, seed_shift=300)
                    
                    def mix_grain(master, var, strength):
                        mixed = master * (1.0 - color_decoherence) + var * color_decoherence
                        return mixed * strength * sens_map

                    raw_gn_r = mix_grain(gn_master, gn_var_r, g_str[0])
                    raw_gn_g = mix_grain(gn_master, gn_var_g, g_str[1])
                    raw_gn_b = mix_grain(gn_master, gn_var_b, g_str[2])

                    # Dye Crosstalk (Grain from one layer affecting others)
                    xtalk_val = p["grain_base"][3]
                    if xtalk_val > 0:
                        xtalk_amount = xtalk_val * grain_factor * sens_map * 0.5
                        gn_r = raw_gn_r + (raw_gn_g + raw_gn_b) * xtalk_amount
                        gn_g = raw_gn_g + (raw_gn_r + raw_gn_b) * xtalk_amount
                        gn_b = raw_gn_b + (raw_gn_r + raw_gn_g) * xtalk_amount
                    else:
                        gn_r, gn_g, gn_b = raw_gn_r, raw_gn_g, raw_gn_b
                else:
                    # B&W Grain
                    gn_l = self.generate_grain(lux_total, g_str[3] * sens_map, 0)

            # --- 4. Composition & Tone Mapping ---
            final_r, final_g, final_b = None, None, None

            if p["type"] == "color":
                bloom_r = apply_bloom(lux_r, 55)
                bloom_g = apply_bloom(lux_g, 35)
                bloom_b = apply_bloom(lux_b, 15)

                dr, lr, xr = p["opt_r"]
                dg, lg, xg = p["opt_g"]
                db, lb, xb = p["opt_b"]
                
                # Optical Mix
                l_r_comp = bloom_r * dr + (lux_r**xr) * lr + gn_r
                l_g_comp = bloom_g * dg + (lux_g**xg) * lg + gn_g
                l_b_comp = bloom_b * db + (lux_b**xb) * lb + gn_b
                
                # Local Contrast Enhancement (Clarity)
                # Prevents flat look from compression
                base_lce_strength  = 0.15 * p["sens_factor"]

                # Create a highlight mask: 0 in shadows, 1 in pure white
                # This mask will control how much LCE is applied to different brightness levels
                # LCE will be stronger in brighter areas (0.7 to 1.0)
                highlight_mask = np.clip(avg_lux_map * 4.0 - 2.8, 0.0, 1.0) # Curves for highlight emphasis (adjust 4.0, -2.8 as needed)
                
                # Blend LCE strength: base_lce_strength + additional strength for highlights
                # Example: Highlights get base + (base * 0.5) = 1.5 * base
                effective_lce_strength = base_lce_strength + (base_lce_strength * 0.5 * highlight_mask)
                
                # Apply LCE per channel using the highlight-weighted strength map
                final_r = self.apply_local_contrast(l_r_comp, effective_lce_strength)
                final_g = self.apply_local_contrast(l_g_comp, effective_lce_strength)
                final_b = self.apply_local_contrast(l_b_comp, effective_lce_strength)

                if tone_mapping == "filmic":
                    final_r = self.filmic_curve(l_r_comp, p["curve"], exposure)
                    final_g = self.filmic_curve(l_g_comp, p["curve"], exposure)
                    final_b = self.filmic_curve(l_b_comp, p["curve"], exposure)
                    
                    # --- Highlight Saturation Compensation ---
                    # 1. Boost saturation slightly to recover density lost by S-Curve
                    merged_temp = cv2.merge([final_r, final_g, final_b])
                    merged_temp = self.adjust_saturation(merged_temp, 1.15)
                    
                    # 2. Highlight Protection (Only for pure whites > 0.95)
                    luma_final = 0.2126*merged_temp[:,:,0] + 0.7152*merged_temp[:,:,1] + 0.0722*merged_temp[:,:,2]
                    luma_final = np.stack([luma_final]*3, axis=2)
                    
                    highlight_mask = np.clip((luma_final - 0.95) * 20.0, 0.0, 1.0)
                    merged_final = merged_temp * (1.0 - highlight_mask) + luma_final * highlight_mask
                    
                    final_r, final_g, final_b = cv2.split(merged_final)

                elif tone_mapping == "reinhard":
                    gamma_param = p["curve"]["gamma"]
                    final_r = self.reinhard_curve(l_r_comp, gamma_param)
                    final_g = self.reinhard_curve(l_g_comp, gamma_param)
                    final_b = self.reinhard_curve(l_b_comp, gamma_param)
                else:
                    final_r, final_g, final_b = l_r_comp, l_g_comp, l_b_comp
                
                # Split Toning
                tint_data = p.get("tint", {})
                final_r, final_g, final_b = self.apply_split_toning(final_r, final_g, final_b, tint_data)
                merged = cv2.merge([final_r, final_g, final_b])

            else: # B&W Logic
                bloom_l = apply_bloom(lux_total, 55)
                if grain_factor <= 0: gn_l = 0
                
                dl, ll, xl = p["opt_l"]
                l_total_comp = bloom_l * dl + (lux_total**xl) * ll + gn_l
                
                # [REVISED] Apply Local Contrast, with Highlight Emphasis for B&W
                base_lce_strength = 0.20 * p["sens_factor"] # Higher base for B&W
                
                highlight_mask = np.clip(avg_lux_map * 4.0 - 2.8, 0.0, 1.0)
                effective_lce_strength = base_lce_strength + (base_lce_strength * 0.5 * highlight_mask)
                
                l_total_comp = self.apply_local_contrast(l_total_comp, effective_lce_strength)
                
                if tone_mapping == "filmic":
                    final_bw = self.filmic_curve(l_total_comp, p["curve"], exposure)
                elif tone_mapping == "reinhard":
                    final_bw = self.reinhard_curve(l_total_comp, p["curve"]["gamma"])
                else:
                    final_bw = l_total_comp
                
                tint_data = p.get("tint", {})
                fr, fg, fb = final_bw, final_bw, final_bw
                fr, fg, fb = self.apply_split_toning(fr, fg, fb, tint_data)
                merged = cv2.merge([fr, fg, fb])

            merged = np.clip(merged, 0, 1)
            output_images.append(torch.from_numpy(merged))

        return (torch.stack(output_images),)
    
NODE_CLASS_MAPPINGS = {
    "FilmSimNode": FilmSimNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FilmSimNode": "Film Simulation V4 (Ultimate)"
}
