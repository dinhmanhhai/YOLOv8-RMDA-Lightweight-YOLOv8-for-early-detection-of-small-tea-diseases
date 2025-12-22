"""
Định nghĩa Conv tuỳ chỉnh `ConvRFC` và đăng ký cho YOLO sử dụng trong YAML.
"""

import torch.nn as nn
from ultralytics.nn.modules import Conv as BaseConv
import builtins
import ultralytics.nn.tasks as tasks_module
import ultralytics.nn.modules as modules


class ConvRFC(BaseConv):
    """
    Conv tuỳ chỉnh dựa trên Conv chuẩn của Ultralytics.
    Signature giống hệt Conv: (c1, c2, k=1, s=1, p=None, g=1, d=1, act=True)
    """
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize ConvRFC với signature giống Conv chuẩn.
        
        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size
            s: Stride
            p: Padding
            g: Groups
            d: Dilation
            act: Activation
        """
        # Gọi __init__ của BaseConv với đúng tham số
        super().__init__(c1, c2, k=k, s=s, p=p, g=g, d=d, act=act)

def register_yolo_modules() -> None:
    """
    Đăng ký ConvRFC vào không gian tên Ultralytics:
    - ultralytics.nn.modules (modules.ConvRFC)
    - ultralytics.nn.tasks (tasks.ConvRFC)
    - parse_model.__globals__ (để eval tìm thấy)
    Gọi hàm này TRƯỚC KHI tạo YOLO(...).
    """

    # Đăng ký trong modules
    modules.ConvRFC = ConvRFC

    # Đăng ký trong tasks
    setattr(tasks_module, "ConvRFC", ConvRFC)

    # Đăng ký trong parse_model globals
    if hasattr(tasks_module, "parse_model"):
        g = tasks_module.parse_model.__globals__
        g["ConvRFC"] = ConvRFC

    # Fallback trong builtins
    setattr(builtins, "ConvRFC", ConvRFC)


