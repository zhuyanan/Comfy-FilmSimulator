import torch
import numpy as np
import os
import cv2
import json

# ==============================================================================
# 1. 预设加载
# ==============================================================================
def load_presets():
    try:
        current_dir = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        current_dir = os.getcwd()
    json_path = os.path.join(current_dir, "films.json")
    
    default_presets = {
        "Kodak Portra 400": {
            "type": "color",
            "matrix": [1.08, -0.04, -0.04, -0.02, 1.05, -0.03, -0.01, -0.03, 1.04, 0, 0, 0],
            "sens_factor": 1.0,
            "grain_base": [0.15, 0.15, 0.15, 0.05],
            "opt_r": [1.15, 1.00, 1.10], # [Diffusion, Linear, Gamma]
            "opt_g": [1.05, 1.00, 1.05], 
            "opt_b": [1.05, 0.97, 1.00],
            "tint": {"shadows": [0.0, 0.0, 0.02], "highlights": [0.02, 0.01, 0.0]}
        }
    }

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading films.json: {e}")
            pass
    return default_presets

FILM_PRESETS = load_presets()

# ==============================================================================
# 2. 胶片模拟节点
# ==============================================================================
class FilmSimNode:
    def __init__(self):
        self.time_hash = 0

    @classmethod
    def INPUT_TYPES(s):
        film_list = list(FILM_PRESETS.keys())
        default_film = film_list[0] if film_list else "Kodak Portra 400"
        return {
            "required": {
                "image": ("IMAGE",),
                "film_type": (film_list, {"default": default_film}),
                
                "wb_temperature_K": ("INT", {"default": 5600, "min": 2000, "max": 12000, "step": 50}),
                "wb_tint": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 50.0, "step": 0.1}),
                
                "auto_exposure": ("BOOLEAN", {"default": True, "label_on": "Auto (Standard 18%)", "label_off": "Manual (Raw Input)"}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1}),
                
                "effect_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "grain_power": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "halation_power": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "local_contrast": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "is_linear_input": ("BOOLEAN", {"default": True})
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("hdr_image", "preview_sdr")
    FUNCTION = "apply_film"
    CATEGORY = "SmartHDR"

    def get_film_params(self, name):
        p = FILM_PRESETS.get(name, {"type":"color"}).copy()
        if "matrix" not in p or p["matrix"] is None: p["matrix"] = [1,0,0,0, 1,0,0,0, 1,0,0,0]
        while len(p["matrix"]) < 12: p["matrix"].append(0.0)
        if "grain_base" not in p: p["grain_base"] = [0,0,0,0]
        while len(p["grain_base"]) < 4: p["grain_base"].append(0.0)
        if "sens_factor" not in p: p["sens_factor"] = 1.0
        # Default [Diffusion, Linear, Gamma]
        if "opt_r" not in p: p["opt_r"] = [1.0, 1.0, 1.0]; p["opt_g"] = [1.0, 1.0, 1.0]; p["opt_b"] = [1.0, 1.0, 1.0]
        if "opt_l" not in p: p["opt_l"] = [1.0, 1.0, 1.0]
        return p

    def calculate_auto_exposure(self, img_np, target_gray=0.18):
        luma = 0.2126 * img_np[:,:,0] + 0.7152 * img_np[:,:,1] + 0.0722 * img_np[:,:,2]
        valid_mask = luma > 0.001
        if np.sum(valid_mask) > 0:
            valid_luma = luma[valid_mask]
            log_avg = np.mean(np.log(valid_luma))
            avg_luma = np.exp(log_avg)
        else:
            avg_luma = 0.18
        scale = target_gray / avg_luma
        scale = np.clip(scale, 0.1, 1000.0)
        return scale

    def kelvin_to_rgb(self, k):
        temp = k / 100.0
        r, g, b = 0, 0, 0
        if temp <= 66:
            r = 255
            g = 99.4708025861 * np.log(temp) - 161.1195681661
            if temp <= 19: b = 0
            else: b = 138.5177312231 * np.log(temp - 10) - 305.0447927307
        else:
            r = 329.698727446 * ((temp - 60) ** -0.1332047592)
            g = 288.1221695283 * ((temp - 60) ** -0.0755148492)
            b = 255
        return max(0,min(255,r))/255.0, max(0,min(255,g))/255.0, max(0,min(255,b))/255.0

    def apply_physical_white_balance(self, img, kelvin, tint):
        light_r, light_g, light_b = self.kelvin_to_rgb(kelvin)
        ref_r, ref_g, ref_b = self.kelvin_to_rgb(5600)
        gain_r = ref_r / max(light_r, 1e-5)
        gain_g = ref_g / max(light_g, 1e-5)
        gain_b = ref_b / max(light_b, 1e-5)
        tint_factor = 1.0 - (tint / 100.0)
        gain_g *= tint_factor
        gain_luma = 0.2126*gain_r + 0.7152*gain_g + 0.0722*gain_b
        norm = 1.0 / max(gain_luma, 1e-5)
        img[:,:,0] *= gain_r * norm
        img[:,:,1] *= gain_g * norm
        img[:,:,2] *= gain_b * norm
        return img

    def apply_local_contrast(self, image, strength=0.2):
        if strength <= 0: return image
        log_img = np.log1p(image)
        h, w = image.shape[:2]
        blur_radius = int(min(h, w) * 0.01) | 1
        blurred_log = cv2.GaussianBlur(log_img, (blur_radius, blur_radius), 0)
        details = log_img - blurred_log
        enhanced_log = log_img + details * strength
        return np.expm1(enhanced_log)

    def log_contrast_curve(self, x, scale, gamma, pivot=0.18):
        """
        Log Pivot Contrast
        Scale: Index 1 (Linear Gain)
        Gamma: Index 2 (Contrast)
        """
        x = np.maximum(x, 1e-6)
        log_x = np.log2(x)
        log_pivot = np.log2(pivot)
        log_out = (log_x - log_pivot) * gamma + log_pivot
        out = np.power(2.0, log_out)
        out = out * scale 
        return out

    def apply_film(self, image, film_type, wb_temperature_K, wb_tint, auto_exposure, exposure, effect_strength, grain_power, halation_power, local_contrast, is_linear_input):
        self.time_hash = int(cv2.getTickCount())
        p = self.get_film_params(film_type)
        
        hdr_results = []
        sdr_preview_results = []
        
        for i in range(image.shape[0]):
            img_np = image[i].cpu().numpy()
            
            # 1. 线性化
            if not is_linear_input:
                img_np = np.power(np.maximum(img_np, 0), 2.2)
            img_np = np.nan_to_num(img_np, nan=0.0, posinf=100.0)
            
            # 2. 白平衡
            img_np = self.apply_physical_white_balance(img_np, wb_temperature_K, wb_tint)
            
            # 3. 曝光
            base_scale = 1.0
            if auto_exposure:
                base_scale = self.calculate_auto_exposure(img_np, target_gray=0.18)
            
            total_scale = base_scale * (2.0 ** exposure)
            img_np = img_np * total_scale
            img_in = img_np.copy()
            height, width = img_np.shape[:2]

            # 4. Matrix (3x3) 转换
            m = p["matrix"]
            if p["type"] == "color":
                r = m[0]*img_np[:,:,0] + m[1]*img_np[:,:,1] + m[2]*img_np[:,:,2]
                g = m[3]*img_np[:,:,0] + m[4]*img_np[:,:,1] + m[5]*img_np[:,:,2]
                b = m[6]*img_np[:,:,0] + m[7]*img_np[:,:,1] + m[8]*img_np[:,:,2]
                
                # 在 HDR 中，Matrix Offset (-0.02) 是相对于中灰的微调
                # 为了防止黑位死黑，我们衰减偏移量
                # hdr_offset_scale = 0.01 
                # r += m[3] * hdr_offset_scale
                # g += m[7] * hdr_offset_scale
                # b += m[11] * hdr_offset_scale
            else:
                l = m[0]*img_np[:,:,0] + m[1]*img_np[:,:,1] + m[2]*img_np[:,:,2]
                # 黑白 Offset
                # l += m[3] * 0.01
                r = g = b = l

            r = np.maximum(r, 0)
            g = np.maximum(g, 0)
            b = np.maximum(b, 0)
            img_graded = np.dstack((r, g, b))

            # 5. Local Contrast
            if local_contrast > 0:
                base_lce = 0.15 * p["sens_factor"] * local_contrast
                img_graded = self.apply_local_contrast(img_graded, base_lce)

            # 6. Halation
            luma = 0.2126*img_graded[:,:,0] + 0.7152*img_graded[:,:,1] + 0.0722*img_graded[:,:,2]
            if p["sens_factor"] > 0 and halation_power > 0:
                threshold_mask = np.maximum(luma - 0.5, 0)
                bloom_mask = threshold_mask / (1.0 + threshold_mask)
                scale_ratio = max(height, width) / 2000.0
                ksize = int(15 * p["sens_factor"] * 2.0 * scale_ratio) * 2 + 1
                intensity = 0.15 * p["sens_factor"] * halation_power
                
                def bloom(c):
                    b = cv2.GaussianBlur(c * bloom_mask, (ksize, ksize), 0)
                    return c + b * intensity
                
                img_graded[:,:,0] = bloom(img_graded[:,:,0])
                img_graded[:,:,1] = bloom(img_graded[:,:,1])
                img_graded[:,:,2] = bloom(img_graded[:,:,2])

            # --- 7. Curve (修正参数索引) ---
            pivot = 0.18
            if p["type"] == "color":
                # JSON Order: [Diffusion, Linear(Scale), Gamma]
                # Index 0: Diffusion (忽略，因为 Halation 步骤已处理)
                # Index 1: Linear Scale (正确使用)
                # Index 2: Gamma
                
                s_r, _, g_r = p["opt_r"]
                s_g, _, g_g = p["opt_g"]
                s_b, _, g_b = p["opt_b"]
                
                img_graded[:,:,0] = self.log_contrast_curve(img_graded[:,:,0], s_r, g_r, pivot)
                img_graded[:,:,1] = self.log_contrast_curve(img_graded[:,:,1], s_g, g_g, pivot)
                img_graded[:,:,2] = self.log_contrast_curve(img_graded[:,:,2], s_b, g_b, pivot)
            else:
                s_l, _, g_l = p["opt_l"]
                luma_bw = np.mean(img_graded, axis=2)
                luma_bw = self.log_contrast_curve(luma_bw, s_l, g_l, pivot)
                img_graded[:,:,0] = img_graded[:,:,1] = img_graded[:,:,2] = luma_bw

            # 8. Grain
            grain_base = p["grain_base"]
            if np.sum(grain_base) > 0 and grain_power > 0:
                np.random.seed(self.time_hash % 2**32)
                luma = np.mean(img_graded, axis=2)
                log_luma = np.log2(np.maximum(luma, 1e-4))
                mask = 1.0 - np.abs(log_luma - np.log2(0.18)) / 5.0
                mask = np.clip(mask, 0.2, 1.0)
                mask = np.expand_dims(mask, -1)
                GLOBAL_GRAIN_SCALE = 0.15 
                if p["type"] == "color":
                    noise_r = np.random.normal(0, 1, (height, width)).astype(np.float32)
                    noise_g = np.random.normal(0, 1, (height, width)).astype(np.float32)
                    noise_b = np.random.normal(0, 1, (height, width)).astype(np.float32)
                    gn_r = noise_r * grain_base[0]
                    gn_g = noise_g * grain_base[1]
                    gn_b = noise_b * grain_base[2]
                    xtalk = grain_base[3]
                    if xtalk > 0:
                        xtalk_amount = xtalk * 0.5
                        gn_r_mix = gn_r + (gn_g + gn_b) * xtalk_amount
                        gn_g_mix = gn_g + (gn_r + gn_b) * xtalk_amount
                        gn_b_mix = gn_b + (gn_r + gn_g) * xtalk_amount
                    else:
                        gn_r_mix, gn_g_mix, gn_b_mix = gn_r, gn_g, gn_b
                    noise_stack = np.dstack((gn_r_mix, gn_g_mix, gn_b_mix))
                    img_graded += noise_stack * grain_power * GLOBAL_GRAIN_SCALE * mask
                else:
                    bw_grain_str = grain_base[3] if len(grain_base) > 3 and grain_base[3] > 0 else grain_base[0]
                    noise_1ch = np.random.normal(0, 1, (height, width, 1)).astype(np.float32)
                    img_graded += noise_1ch * bw_grain_str * grain_power * GLOBAL_GRAIN_SCALE * mask

            # 9. Split Toning
            tint_param = p.get("tint", {})
            if tint_param:
                s_rgb = tint_param.get("shadows", [0,0,0])
                h_rgb = tint_param.get("highlights", [0,0,0])
                luma_tint = np.mean(img_graded, axis=2)
                log_l = np.log2(np.maximum(luma_tint, 1e-4))
                dist = (log_l - np.log2(0.18))
                h_mask = 1.0 / (1.0 + np.exp(-2.0 * dist)) 
                s_mask = 1.0 - h_mask
                s_mask = np.expand_dims(s_mask, -1)
                h_mask = np.expand_dims(h_mask, -1)
                luma_broadcast = np.expand_dims(luma_tint, -1)
                tint_s = np.array(s_rgb).reshape(1,1,3) * 5.0
                tint_h = np.array(h_rgb).reshape(1,1,3) * 5.0
                img_graded += (tint_s * s_mask + tint_h * h_mask) * luma_broadcast

            # 10. Blend
            hdr_final = img_in * (1.0 - effect_strength) + img_graded * effect_strength
            hdr_final = np.maximum(hdr_final, 0.0)
            
            # Black Point Fix
            black_level = np.percentile(hdr_final, 1.0)
            if black_level > 0:
                hdr_final = np.maximum(hdr_final - black_level, 0.0)
            
            hdr_final = np.nan_to_num(hdr_final, nan=0.0, posinf=100.0)

            hdr_results.append(torch.from_numpy(hdr_final))
            
            # 11. SDR Preview
            if "curve" in p:
                # 预览时使用 Filmic 曲线模拟最终调性
                A = p["curve"].get("A", 0.15)
                B = p["curve"].get("B", 0.50)
                C = p["curve"].get("C", 0.10)
                D = p["curve"].get("D", 0.20)
                E = p["curve"].get("E", 0.02)
                F = p["curve"].get("F", 0.30)
                gamma = p["curve"].get("gamma", 2.0)
                
                x = np.maximum(hdr_final, 0)
                x_g = 10.0 * (x ** gamma) 
                sdr_view = ((x_g * (A * x_g + C * B) + D * E) / (x_g * (A * x_g + B) + D * F)) - E/F
            else:
                sdr_view = hdr_final / (1.0 + hdr_final)
                sdr_view = np.power(sdr_view, 1.0/2.2)
                
            sdr_view = np.clip(sdr_view, 0, 1)
            sdr_preview_results.append(torch.from_numpy(sdr_view))
            
        return (torch.stack(hdr_results), torch.stack(sdr_preview_results))