from .dng_reader import DNGImageReader
from .film_sim import FilmSimNode
from .save_avif_hdr import SaveAVIF_HDR  # 注意这里的 .save_avif_hdr 必须对应文件名

NODE_CLASS_MAPPINGS = {
    "DNG Image Reader": DNGImageReader,
    "FilmSimNode": FilmSimNode,
    "SaveAVIF_HDR": SaveAVIF_HDR
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DNG Image Reader": "DNG Image Reader (Rawpy)",
    "FilmSimNode": "Film Simulation V4.1 (HDR Capable)",
    "SaveAVIF_HDR": "Save AVIF/HEIC HDR (Native)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']