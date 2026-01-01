"""
ComfyUI DNG Image Reader自定义节点
该节点能够读取DNG图像文件，进行解码，并输出为Save image或Preview image节点可以接受的格式
"""

import rawpy
import numpy as np
import os
import torch

class DNGImageReader:
    """
    ComfyUI自定义节点类：DNG Image Reader
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点的输入类型
        """
        return {
            "required": {
                "dng_path": ("STRING", {"default": "", "multiline": False}),
                "linear_output": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")  # 返回图像数据和元数据
    RETURN_NAMES = ("image", "metadata")
    FUNCTION = "read_dng"
    CATEGORY = "image"

    def read_dng(self, dng_path, linear_output):
        """
        读取DNG文件并解码
        :param dng_path: DNG文件路径
        :param linear_output: 是否输出16位线性图像
        :return: 解码后的图像数据和元数据
        """
        # 检查文件是否存在
        if not os.path.exists(dng_path):
            raise FileNotFoundError(f"DNG file not found: {dng_path}")
        
        # 检查文件扩展名是否为DNG
        # if not dng_path.lower().endswith('.dng'):
        #    raise ValueError(f"File is not a DNG format: {dng_path}")
        
        try:
            # 使用rawpy读取DNG文件
            with rawpy.imread(dng_path) as raw:
                # 根据linear_output参数决定处理方式
                if linear_output:
                    # 输出16位线性图像 - 不应用伽马校正
                    rgb_image = raw.postprocess(
                        gamma=(1.0, 1.0),  # 线性输出，不应用伽马校正
                        no_auto_bright=True,  # 不自动调整亮度
                        output_bps=16  # 输出16位
                    )
                    # 对于线性输出，保持16位值范围 [0, 65535]，但转换为浮点范围 [0.0, 1.0]
                    # 为了保持线性特性，我们仍然将其归一化到 [0.0, 1.0] 但保留线性关系
                    rgb_image = np.clip(rgb_image, 0, 65535)
                    image_array = rgb_image.astype(np.float32) / 65535.0
                else:
                    # 默认输出 - 应用伽马校正
                    rgb_image = raw.postprocess(
                        gamma=(2.2, 4.5),  # 应用伽马校正
                        no_auto_bright=True,  # 不自动调整亮度
                        output_bps=16  # 输出16位
                    )
                    
                    # 确保值在正确范围内 [0, 65535] -> [0, 1]
                    rgb_image = np.clip(rgb_image, 0, 65535)
                    image_array = rgb_image.astype(np.float32) / 65535.0
                
                # 确保图像是3维张量 (H, W, C)
                if len(image_array.shape) == 3:
                    # 图像已经是 (H, W, C) 格式
                    pass
                elif len(image_array.shape) == 2:
                    # 灰度图扩展为 (H, W, 1) 然后复制为 (H, W, 3)
                    image_array = np.stack([image_array] * 3, axis=-1)
                
                # 转换为PyTorch张量，这是ComfyUI期望的格式
                image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # 添加批次维度 (1, H, W, C)
                
                # 获取图像元数据
                metadata = self.extract_metadata(raw)
                
                return (image_tensor, metadata)
                
        except Exception as e:
            raise Exception(f"Error processing DNG file: {str(e)}")

    def extract_metadata(self, raw):
        """
        提取DNG文件的元数据
        :param raw: rawpy.RawPy对象
        :return: 元数据字符串
        """
        try:
            # 获取基本元数据
            sizes = raw.sizes
            width = sizes.width
            height = sizes.height
            
            # 获取滤镜模式
            filters = getattr(sizes, 'filters', None)
            if filters is not None:
                # 将滤镜模式转换为可读格式
                if filters == 0:
                    bayer_pattern_str = "No filter (Monochrome)"
                elif filters == 9:  # RGGB
                    bayer_pattern_str = "RGGB"
                elif filters == 273:  # GRBG
                    bayer_pattern_str = "GRBG"
                elif filters == 1536:  # BGGR
                    bayer_pattern_str = "BGGR"
                elif filters == 6144:  # GBRG
                    bayer_pattern_str = "GBRG"
                else:
                    bayer_pattern_str = f"Unknown ({filters})"
            else:
                bayer_pattern_str = "Unknown"
            
            # 获取rawpy中可用的属性
            color_desc = raw.color_desc.decode('utf-8') if hasattr(raw.color_desc, 'decode') else str(raw.color_desc)
            
            metadata_str = f"""DNG Metadata:
Width: {width}
Height: {height}
Bayer Pattern: {bayer_pattern_str}
Color Description: {color_desc}"""
            
            # 尝试获取相机信息
            try:
                camera_model = raw.camera_model.decode('utf-8') if raw.camera_model else 'Unknown'
                camera_make = raw.camera_make.decode('utf-8') if raw.camera_make else 'Unknown'
                metadata_str += f"""
Camera: {camera_model}
Camera Maker: {camera_make}"""
            except:
                metadata_str += f"""
Camera: Unknown
Camera Maker: Unknown"""
            
            # 尝试获取其他元数据
            try:
                # 获取黑电平
                black_level = getattr(raw, 'black_level', 'Unknown')
                metadata_str += f"\nBlack Level: {black_level}"
                
                # 获取白电平
                white_level = getattr(raw, 'whitepoint', 'Unknown')
                metadata_str += f"\nWhite Level: {white_level}"
            except:
                pass
            
            # 添加线性输出相关信息
            metadata_str += f"\nLinear Output Mode: {'Enabled' if True else 'Disabled'}"
            
            return metadata_str
        except Exception as e:
            return f"Error extracting metadata: {str(e)}"


# 注册节点
NODE_CLASS_MAPPINGS = {
    "DNG Image Reader": DNGImageReader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DNG Image Reader": "DNG Image Reader"
}



