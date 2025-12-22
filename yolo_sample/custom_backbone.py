"""
Định nghĩa Conv tuỳ chỉnh `ConvRFC` và đăng ký cho YOLO sử dụng trong YAML.
"""

from ultralytics.nn.modules import Conv as BaseConv
import builtins
import ultralytics.nn.tasks as tasks_module
import ultralytics.nn.modules as modules


class ConvRFC(BaseConv):
    """
    Conv tuỳ chỉnh dựa trên Conv chuẩn của Ultralytics.
    Hiện tại giữ nguyên hành vi, bạn có thể mở rộng thêm nếu muốn.
    """

    # Có thể override __init__ hoặc forward nếu cần logic khác.
    pass


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

