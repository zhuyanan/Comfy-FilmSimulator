import os
import numpy as np
import cv2
import torch

class SaveSDR:
    def __init__(self):
        try:
            import folder_paths
            self.folder_paths = folder_paths
            self.output_dir = folder_paths.get_output_directory()
        except Exception:
            self.folder_paths = None
            self.output_dir = os.getcwd()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sdr_image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "SDR_Image"}),
                "format": (["PNG", "JPEG"], {"default": "PNG"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "apply_srgb_gamma": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_sdr"
    OUTPUT_NODE = True
    CATEGORY = "SmartHDR"

    def _to_hwc(self, img):
        if isinstance(img, torch.Tensor):
            arr = img.cpu().numpy()
        else:
            arr = np.array(img)
        if arr.ndim == 3 and arr.shape[0] in (1,3) and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1,2,0))
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        return arr.astype(np.float32)

    def save_sdr(self, sdr_image, filename_prefix, format, quality, apply_srgb_gamma):
        results = []
        try:
            if self.folder_paths is not None:
                full_output_folder, filename, counter, subfolder, filename_prefix = \
                    self.folder_paths.get_save_image_path(filename_prefix, self.output_dir, 512, 512)
            else:
                full_output_folder = os.path.join(self.output_dir, "outputs")
                os.makedirs(full_output_folder, exist_ok=True)
                filename = filename_prefix
                counter = 0
        except Exception:
            full_output_folder = os.path.join(self.output_dir, "outputs")
            os.makedirs(full_output_folder, exist_ok=True)
            filename = filename_prefix
            counter = 0

        for img in sdr_image:
            try:
                img_np = self._to_hwc(img)
                # Clip and sanitize
                img_np = np.nan_to_num(img_np, nan=0.0, posinf=1e6, neginf=0.0)
                img_np = np.maximum(img_np, 0.0)

                # If requested, convert linear->sRGB with gamma ~2.2
                out = img_np.copy()
                if apply_srgb_gamma:
                    out = np.power(np.clip(out, 0.0, 1.0), 1.0/2.2)

                out8 = (np.clip(out, 0.0, 1.0) * 255.0).round().astype(np.uint8)

                ext = ".png" if format == "PNG" else ".jpg"
                f_path = os.path.join(full_output_folder, f"{filename}_{counter:05}{ext}")

                # cv2 expects BGR
                bgr = cv2.cvtColor(out8, cv2.COLOR_RGB2BGR)
                if format == "PNG":
                    success = cv2.imwrite(f_path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                else:
                    success = cv2.imwrite(f_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])

                if success:
                    print(f"Saved SDR: {f_path}")
                    results.append({"filename": f_path, "type": "output"})
                else:
                    print(f"[SaveSDR] Failed to write {f_path}")
                counter += 1
            except Exception as e:
                print(f"[SaveSDR] Error saving one image: {e}")
                continue

        return {"ui": {"images": results}}
