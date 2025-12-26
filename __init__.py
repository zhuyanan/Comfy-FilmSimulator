import torch
import numpy as np
import json
import os
import sys

# --- 依赖检查 ---
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
    
    # 默认回退预设
    default_presets = {
        "NC200 (Fuji C200 style)": {
            "type": "color",
            "matrix": [0.77, 0.12, 0.18, 0.08, 0.85, 0.23, 0.08, 0.09, 0.92, 0.25, 0.35, 0.35],
            "sens_factor": 1.20,
            "grain_base": [0.18, 0.18, 0.18, 0.08],
            "opt_r": [1.0, 1.0, 1.0],
            "opt_g": [1.0, 1.0, 1.0],
            "opt_b": [1.0, 1.0, 1.0],
            "curve": {"A": 0.15, "B": 0.50, "C": 0.10, "D": 0.20, "E": 0.02, "F": 0.30, "gamma": 2.05, "exposure_bias": 0.0}
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
    Film Simulation for ComfyUI (Adaptive V3)
    - Adaptive Mid-Gray Anchor: Brightness doesn't drift with Gamma.
    - Matte Look: Black/White point mapped to 0.02 - 0.98.
    - Exposure Control: Node parameter + JSON bias.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        film_list = list(FILM_PRESETS.keys())
        default_film = "NC200 (Fuji C200 style)"
        if default_film not in film_list and len(film_list) > 0:
            default_film = film_list[0]

        return {
            "required": {
                "image": ("IMAGE",),
                "film_type": (film_list, {"default": default_film}),
                "tone_mapping": (["filmic", "reinhard", "none"], {"default": "filmic"}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1, "tooltip": "Manual exposure compensation in EV"}),
                "grain_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "halation_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_film"
    CATEGORY = "FilmSim/Film"

    def get_film_params(self, film_type_name):
        p = FILM_PRESETS.get(film_type_name)
        if p is None:
             p = list(FILM_PRESETS.values())[0]
        
        # 确保参数完整性
        if p["type"] == "color":
             if "opt_r" not in p: p["opt_r"] = [1.0, 1.0, 1.0]
             if "opt_g" not in p: p["opt_g"] = [1.0, 1.0, 1.0]
             if "opt_b" not in p: p["opt_b"] = [1.0, 1.0, 1.0]
        elif p["type"] == "bw":
             if "opt_l" not in p: p["opt_l"] = [1.0, 1.0, 1.0]
             
        # 确保 exposure_bias 存在
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
        if channel is None: return None
        np.random.seed(int(self.time_hash + seed_shift) % 2**32)
        
        noise = np.random.normal(0, 1, channel.shape).astype(np.float32)
        noise = noise ** 2
        noise = noise * (np.random.choice([-1, 1], channel.shape))
        
        weights = (0.5 - np.abs(channel - 0.5)) * 2
        weights = np.clip(weights, 0.05, 0.9)
        
        weighted_noise = noise * weights * grain_intensity
        weighted_noise = cv2.GaussianBlur(weighted_noise, (3, 3), 1)
        return np.clip(weighted_noise, -1, 1)

    def filmic_curve(self, x, c, manual_exposure=0.0):
        if x is None: return None
        x = np.maximum(x, 0)
        
        # --- 1. 自适应增益计算 (Adaptive Gain) ---
        # 目的：锚定中灰点。无论 Gamma 是多少，输入 0.18 的亮度响应保持为基准值。
        # 基准：Gamma=2.0, Gain=10.0 (经典 Hable 设定)
        base_scaling = 10.0 * (0.18 ** 2.0)
        current_mid_response = 0.18 ** c["gamma"]
        # +1e-6 防止除零
        standard_adaptive_scale = base_scaling / (current_mid_response + 1e-6)
        
        # --- 2. 曝光控制 (Exposure Control) ---
        # 综合曝光 = JSON预设偏差 + 节点手动调节
        json_bias = c.get("exposure_bias", 0.0)
        total_ev = json_bias + manual_exposure
        exposure_mult = 2.0 ** total_ev
        
        # --- 3. 应用增益到图像 ---
        # 图像信号受到曝光影响：变亮或变暗
        curr_x = (standard_adaptive_scale * exposure_mult) * (x ** c["gamma"]) 
        
        A, B, C, D, E, F = c["A"], c["B"], c["C"], c["D"], c["E"], c["F"]
        
        def curve_fn(t):
            # 修复：分母加 1e-6 防止除零
            denominator = (t * (A * t + B) + D * F) + 1e-6 
            numerator = (t * (A * t + C * B) + D * E)
            return (numerator / denominator) - (E / F)

        mapped = curve_fn(curr_x)
        
        # --- 4. 动态白点 W 计算 (关键逻辑) ---
        # W 仅基于 "标准自适应比例" 计算，不包含 exposure_mult。
        # 这样，"白色的定义" 是固定的。
        # 如果降低曝光，图像信号变小，远离 W，因此看起来变暗（正确模拟欠曝）。
        # 如果 W 也随曝光降低，画面会被重新拉伸回 1.0，导致曝光调整失效。
        
        W_linear = (1.0 / 0.18) ** c["gamma"]
        W_input = W_linear * standard_adaptive_scale
        
        white_scale = curve_fn(np.array([W_input]))
        if white_scale < 1e-4: 
            white_scale = 1.0 
            
        normalized = mapped / white_scale
        
        # --- 5. 哑光胶片感映射 (Matte Look) ---
        # 将输出限制在 0.02 (深灰) 到 0.98 (亮灰) 之间
        # 模拟相纸无法表现绝对黑白的物理特性
        target_black = 0.02
        target_white = 0.98
        
        return normalized * (target_white - target_black) + target_black

    def reinhard_curve(self, x, gamma):
        if x is None: return None
        mapped = x * (x / (1.0 + x))
        mapped = np.power(mapped, 1.0/gamma)
        return np.clip(mapped, 0, 1)

    def process_film(self, image, film_type, tone_mapping, exposure, grain_factor, halation_factor):
        self.time_hash = int(cv2.getTickCount())
        output_images = []
        
        for i in range(image.shape[0]):
            img_np = image[i].cpu().numpy()
            
            height, width = img_np.shape[:2]
            p = self.get_film_params(film_type)
            
            lux_r, lux_g, lux_b, lux_total = self.luminance_calc(img_np, p)
            
            if p["type"] == "color":
                avg_lux = (np.mean(lux_r) + np.mean(lux_g) + np.mean(lux_b)) / 3.0
            else:
                avg_lux = np.mean(lux_total)

            sens = (1.0 - avg_lux) * 0.75 + 0.10
            sens = np.clip(sens, 0.10, 0.7)
            
            scale_ratio = max(height, width) / 3000.0
            rads = int(20 * (sens**2) * p["sens_factor"] * 2.0 * scale_ratio) 
            rads = max(1, rads)
            
            ksize = rads * 2 + 1
            base_diffusion = 0.05 * p["sens_factor"]
            halo_str = 23 * sens**2 * p["sens_factor"] * halation_factor

            def apply_bloom(lux_channel, sigma_mult):
                if lux_channel is None: return None
                weights = (base_diffusion + lux_channel**2) * sens
                weights = np.clip(weights, 0, 1)
                k = int(ksize * 3) | 1
                bloom = cv2.GaussianBlur(lux_channel * weights, (k, k), sens * sigma_mult * scale_ratio)
                effect = bloom * weights * halo_str
                return effect / (1.0 + effect)

            final_r, final_g, final_b = None, None, None
            grain_str = [g * grain_factor * sens for g in p["grain_base"]]

            if p["type"] == "color":
                bloom_r = apply_bloom(lux_r, 55)
                bloom_g = apply_bloom(lux_g, 35)
                bloom_b = apply_bloom(lux_b, 15)
                
                gn_r = self.generate_grain(lux_r, grain_str[0], 0) if grain_factor > 0 else 0
                gn_g = self.generate_grain(lux_g, grain_str[1], 100) if grain_factor > 0 else 0
                gn_b = self.generate_grain(lux_b, grain_str[2], 200) if grain_factor > 0 else 0

                dr, lr, xr = p["opt_r"]
                dg, lg, xg = p["opt_g"]
                db, lb, xb = p["opt_b"]
                
                # Composition
                l_r_comp = bloom_r * dr + (lux_r**xr) * lr
                l_g_comp = bloom_g * dg + (lux_g**xg) * lg
                l_b_comp = bloom_b * db + (lux_b**xb) * lb
                
                # Cross-talk Grain
                cross_talk = p["grain_base"][3] * grain_factor * sens
                l_r_comp += gn_r + gn_g * cross_talk + gn_b * cross_talk
                l_g_comp += gn_r * cross_talk + gn_g + gn_b * cross_talk
                l_b_comp += gn_r * cross_talk + gn_g * cross_talk + gn_b

                if tone_mapping == "filmic":
                    # 传入 exposure 参数
                    final_r = self.filmic_curve(l_r_comp, p["curve"], exposure)
                    final_g = self.filmic_curve(l_g_comp, p["curve"], exposure)
                    final_b = self.filmic_curve(l_b_comp, p["curve"], exposure)
                elif tone_mapping == "reinhard":
                    gamma_param = p["curve"]["gamma"]
                    final_r = self.reinhard_curve(l_r_comp, gamma_param)
                    final_g = self.reinhard_curve(l_g_comp, gamma_param)
                    final_b = self.reinhard_curve(l_b_comp, gamma_param)
                else:
                    final_r, final_g, final_b = l_r_comp, l_g_comp, l_b_comp

                merged = cv2.merge([final_r, final_g, final_b])

            else: # B&W
                bloom_l = apply_bloom(lux_total, 55)
                gn_l = self.generate_grain(lux_total, grain_str[3], 0) if grain_factor > 0 else 0
                
                dl, ll, xl = p["opt_l"]
                l_total_comp = bloom_l * dl + (lux_total**xl) * ll + gn_l
                
                if tone_mapping == "filmic":
                    # 传入 exposure 参数
                    final_bw = self.filmic_curve(l_total_comp, p["curve"], exposure)
                elif tone_mapping == "reinhard":
                    final_bw = self.reinhard_curve(l_total_comp, p["curve"]["gamma"])
                else:
                    final_bw = l_total_comp
                    
                merged = cv2.merge([final_bw, final_bw, final_bw])

            # Safety clip
            merged = np.clip(merged, 0, 1)
            
            output_images.append(torch.from_numpy(merged))

        return (torch.stack(output_images),)
    
NODE_CLASS_MAPPINGS = {
    "FilmSimNode": FilmSimNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FilmSimNode": "Film Simulation V3 (Adaptive)"
}