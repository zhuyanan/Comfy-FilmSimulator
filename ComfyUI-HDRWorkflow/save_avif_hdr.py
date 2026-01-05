import torch
import numpy as np
import os
import folder_paths

HAS_HEIF = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    try: pillow_heif.register_avif_opener()
    except: pass
    HAS_HEIF = True
except: pass

class SaveAVIF_HDR:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hdr_image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "HDR_Image"}),
                "format": (["AVIF", "HEIC"], {"default": "AVIF"}),
                "quality": ("INT", {"default": 90}),
                "ref_white_nits": ("FLOAT", {"default": 203.0, "min": 80.0, "max": 1000.0}), 
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_hdr"
    OUTPUT_NODE = True
    CATEGORY = "SmartHDR"

    def save_hdr(self, hdr_image, filename_prefix, format, quality, ref_white_nits):
        if not HAS_HEIF: return {}

        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir, hdr_image[0].shape[1], hdr_image[0].shape[0])
        results = []
        
        m_2020 = np.array([[0.6274, 0.3293, 0.0433], [0.0690, 0.9196, 0.0114], [0.0164, 0.0880, 0.8956]])

        for img in hdr_image:
            img_np = img.cpu().numpy()
            
            # 1. 彻底的数据清洗
            img_np = np.nan_to_num(img_np, nan=0.0, posinf=100.0, neginf=0.0)
            img_np = np.maximum(img_np, 0.0)

            # 2. Rec.2020 转换
            img_rec2020 = np.dot(img_np, m_2020.T)
            
            # 3. PQ 转换
            nits = img_rec2020 * ref_white_nits
            nits = np.clip(nits, 0, 10000.0)
            y = nits / 10000.0
            
            m1, m2 = 0.1593017578125, 78.84375
            c1, c2, c3 = 0.8359375, 18.8515625, 18.623046875
            
            num = c1 + c2 * np.power(y, m1)
            den = 1.0 + c3 * np.power(y, m1)
            pq_val = np.power(num / den, m2)
            
            # 4. 16-bit 转换 (修复噪点核心)
            # 使用 np.uint16 (小端序/本机序)，不要强制 >u2，让 pillow_heif 处理
            img_16bit = (np.clip(pq_val, 0, 1) * 65535).astype(np.uint16)
            
            # 关键：确保内存连续！否则 bytes() 会读取到垃圾数据
            img_16bit = np.ascontiguousarray(img_16bit)
            
            # 5. 保存
            heif_file = pillow_heif.from_bytes(
                mode="RGB;16", 
                size=(img_np.shape[1], img_np.shape[0]), 
                data=bytes(img_16bit)
            )
            
            ext = ".avif" if format == "AVIF" else ".heic"
            f_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_{ext}")
            
            heif_file.save(f_path, quality=quality, bit_depth=10, 
                           color_primaries=9, transfer_characteristics=16, matrix_coefficients=9, full_range_flag=True)
            
            print(f"Saved HDR: {f_path}")
            results.append({"filename": f_path, "type": "output"})
            counter += 1

        return {"ui": {"images": results}}