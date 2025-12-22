"""
Wrapper đăng ký lại các module backbone tùy chỉnh từ project improved_yolov8
để có thể sử dụng trong YAML của `yolo_sample`.

Ta KHÔNG tự re-implement mà import trực tiếp:
- RFCBAMConv, C2f_RFCBAM, MixSPPF
giống hệt code đang chạy ổn trong `improved_yolov8`.
"""

from improved_yolov8.models.rfcbam import RFCBAMConv
from improved_yolov8.models.blocks import C2f_RFCBAM, MixSPPF


def register_yolo_sample_modules() -> None:
    """
    Đăng ký các module custom vào Ultralytics để parse từ YAML:
    - Thêm vào ultralytics.nn.tasks
    - Thêm vào ultralytics.nn.modules
    - Patch parse_model.__globals__
    Gọi hàm này TRƯỚC khi tạo YOLO(model_yaml).
    """

    import builtins
    import ultralytics.nn.tasks as tasks_module
    import ultralytics.nn.modules as modules

    custom_modules = {
        "RFCBAMConv": RFCBAMConv,
        "C2f_RFCBAM": C2f_RFCBAM,
        "MixSPPF": MixSPPF,
    }

    # Đăng ký trong tasks_module + builtins
    for name, cls in custom_modules.items():
        setattr(tasks_module, name, cls)
        setattr(builtins, name, cls)

    # Đăng ký trong modules (không bắt buộc nhưng tiện)
    for name, cls in custom_modules.items():
        setattr(modules, name, cls)

    # Patch globals của parse_model
    if hasattr(tasks_module, "parse_model"):
        g = tasks_module.parse_model.__globals__
        for name, cls in custom_modules.items():
            g[name] = cls



