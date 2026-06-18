import torch
import numpy as np
import os
import folder_paths
import sys

HAS_HEIF = False
try:
    import pillow_heif
    # register openers where available
    try:
        pillow_heif.register_heif_opener()
    except Exception:
        pass
    try:
        pillow_heif.register_avif_opener()
    except Exception:
        pass
    HAS_HEIF = True
except Exception:
    HAS_HEIF = False

class SaveAVIF_HDR:
    def __init__(self):
        # folder_paths.get_output_directory() may raise if folder_paths not configured; guard lightly
        try:
            self.output_dir = folder_paths.get_output_directory()
        except Exception:
            self.output_dir = os.getcwd()

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
        if not HAS_HEIF:
            print("[SaveAVIF_HDR] pillow_heif not available; cannot save AVIF/HEIC.")
            return {}

        # Derive size -> try to be defensive about input shape/order
        results = []
        # Rec.2020 primaries matrix
        m_2020 = np.array([[0.6274, 0.3293, 0.0433],
                           [0.0690, 0.9196, 0.0114],
                           [0.0164, 0.0880, 0.8956]], dtype=np.float32)

        # prepare output folder and naming via helper; guard against unexpected shapes
        try:
            # hdr_image[0].shape is used by the original code; attempt to read safely
            sample = hdr_image[0]
            sample_np = sample.cpu().numpy()
            if sample_np.ndim == 4:
                # unexpected batch dim again; collapse
                sample_np = sample_np.squeeze()
            if sample_np.ndim == 3 and sample_np.shape[0] in (1, 3) and sample_np.shape[-1] != 3:
                # channels-first (C,H,W) -> to H,W,C
                sample_np = np.transpose(sample_np, (1, 2, 0))
            h, w = sample_np.shape[:2]
        except Exception:
            h, w = (512, 512)

        # obtain path info from folder helper (same call pattern as original)
        try:
            full_output_folder, filename, counter, subfolder, filename_prefix = \
                folder_paths.get_save_image_path(filename_prefix, self.output_dir, w, h)
        except Exception:
            # fallback if helper fails
            full_output_folder = os.path.join(self.output_dir, "outputs")
            os.makedirs(full_output_folder, exist_ok=True)
            filename = filename_prefix
            counter = 0

        os.makedirs(full_output_folder, exist_ok=True)

        for img in hdr_image:
            try:
                img_np = img.cpu().numpy()
                # handle (C,H,W) -> (H,W,C)
                if img_np.ndim == 3 and img_np.shape[0] in (1, 3) and img_np.shape[-1] != 3:
                    img_np = np.transpose(img_np, (1, 2, 0))
                # squeeze unnecessary leading dims
                if img_np.ndim == 4:
                    img_np = img_np.squeeze()
                # ensure float32
                img_np = img_np.astype(np.float32)
                # clamp negatives introduced by processing
                img_np = np.nan_to_num(img_np, nan=0.0, posinf=1e6, neginf=0.0)
                img_np = np.maximum(img_np, 0.0)

                # Rec.2020 conversion (apply matrix to last axis)
                # If image shape is HxWx3, np.dot with m_2020.T gives HxWx3
                img_rec2020 = np.dot(img_np, m_2020.T).astype(np.float32)

                # Basic input range check: warn if values look like they're already in absolute nits
                max_val = float(np.max(img_rec2020)) if img_rec2020.size else 0.0
                if max_val > 1.5:
                    print(f"[SaveAVIF_HDR] WARNING: max Rec.2020 value {max_val:.3f} > 1.5 — input may already be in scene luminance units (nits). Ensure ref_white_nits is correct to avoid over/under exposure in PQ mapping.")

                # PQ mapping: map scene-referred values (relative to ref_white_nits) to PQ normalized range
                # The node expects scene-referred linear values where 1.0 corresponds to '1 unit' and
                # `ref_white_nits` describes the mapping to absolute luminance for PQ encoding.
                nits = img_rec2020 * float(ref_white_nits)
                nits = np.clip(nits, 0.0, 10000.0)
                y = nits / 10000.0

                # SMPTE ST 2084 constants (m1/m2/c1/c2/c3) per standard - used for PQ EOTF
                # Reference: SMPTE ST 2084 (Perceptual Quantizer) constants
                m1, m2 = 0.1593017578125, 78.84375
                c1, c2, c3 = 0.8359375, 18.8515625, 18.623046875

                # avoid negative/zero bases for power
                y_pos = np.maximum(y, 0.0)
                num = c1 + c2 * np.power(y_pos, m1)
                den = 1.0 + c3 * np.power(y_pos, m1)
                # guard divide
                ratio = np.divide(num, den, out=np.zeros_like(num, dtype=np.float32), where=(den != 0))
                pq_val = np.power(np.maximum(ratio, 0.0), m2).astype(np.float32)

                # convert to 16-bit container (0..1 -> 0..65535)
                # ensure correct byte order and contiguous memory
                clipped = np.clip(pq_val, 0.0, 1.0)
                img_16bit = (clipped * 65535.0).round().astype(np.uint16)
                img_16bit = np.ascontiguousarray(img_16bit)

                # Compute HDR metadata (maxCLL, maxFALL) from luminance (nits)
                try:
                    maxcll = int(np.ceil(np.max(nits)))
                    # Use the mean of pixel nits as frame-average (robust: use 99th percentile of per-frame averages if multi-frame)
                    maxfall = int(np.ceil(np.mean(nits)))
                    # clamp to reasonable HEIF limits
                    maxcll = int(np.clip(maxcll, 0, 10000))
                    maxfall = int(np.clip(maxfall, 0, 10000))
                except Exception:
                    maxcll = None
                    maxfall = None

                # pillow_heif.from_bytes expects raw bytes. Use .tobytes() which is well-defined.
                # The mode 'RGB;16' indicates 16-bit per channel; size is (width, height)
                heif_file = None
                try:
                    heif_file = pillow_heif.from_bytes(
                        mode="RGB;16",
                        size=(img_16bit.shape[1], img_16bit.shape[0]),
                        data=img_16bit.tobytes()
                    )
                except Exception as e:
                    # Some systems or pillow_heif builds may expect a different endian ordering for 16-bit.
                    # Try byteswapped 16-bit data before falling back to 8-bit.
                    print(f"[SaveAVIF_HDR] pillow_heif.from_bytes (native 16-bit) failed: {e}")
                    try:
                        swapped = img_16bit.byteswap().tobytes()
                        heif_file = pillow_heif.from_bytes(
                            mode="RGB;16",
                            size=(img_16bit.shape[1], img_16bit.shape[0]),
                            data=swapped
                        )
                        print("[SaveAVIF_HDR] Used byteswapped 16-bit data as a fallback for pillow_heif.from_bytes.")
                    except Exception as e2:
                        print(f"[SaveAVIF_HDR] pillow_heif.from_bytes (byteswapped) also failed: {e2}")
                        # try a fallback: convert to 8-bit (lossy) to ensure at least something is saved
                        try:
                            img_8bit = (clipped * 255.0).round().astype(np.uint8)
                            img_8bit = np.ascontiguousarray(img_8bit)
                            heif_file = pillow_heif.from_bytes(
                                mode="RGB",
                                size=(img_8bit.shape[1], img_8bit.shape[0]),
                                data=img_8bit.tobytes()
                            )
                            print("[SaveAVIF_HDR] WARNING: Falling back to 8-bit (SDR) due to 16-bit encoding limitations.")
                        except Exception as e3:
                            print(f"[SaveAVIF_HDR] fallback to 8-bit also failed: {e3}")
                            continue

                ext = ".avif" if format == "AVIF" else ".heic"
                f_path = os.path.join(full_output_folder, f"{filename}_{counter:05}{ext}")

                # Save with guarded params; pillow_heif.save can raise
                try:
                    save_kwargs = dict(
                        quality=int(quality),
                        bit_depth=10,
                        color_primaries=9,
                        transfer_characteristics=16,
                        matrix_coefficients=9,
                        full_range_flag=True,
                    )
                    # include HDR metadata if computed
                    if maxcll is not None:
                        save_kwargs["maxcll"] = int(maxcll)
                    if maxfall is not None:
                        save_kwargs["maxfall"] = int(maxfall)

                    heif_file.save(f_path, **save_kwargs)
                except TypeError:
                    # some pillow_heif versions may not accept the same kwargs; try without extras
                    try:
                        heif_file.save(f_path, quality=int(quality))
                    except Exception as e:
                        print(f"[SaveAVIF_HDR] Save failed for {f_path}: {e}")
                        continue
                except Exception as e:
                    print(f"[SaveAVIF_HDR] Save failed for {f_path}: {e}")
                    continue

                print(f"Saved HDR: {f_path}")
                results.append({"filename": f_path, "type": "output"})
                counter += 1

            except Exception as e:
                print(f"[SaveAVIF_HDR] Error processing one image: {e}")
                continue

        return {"ui": {"images": results}}
