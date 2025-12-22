#!/usr/bin/env python3
"""
Wrapper để đảm bảo TDDet modules được import trước khi chạy YOLO CLI.
Script này import modules từ TDDet/codes rồi mới chạy yolo command.
"""

import sys
import os
from pathlib import Path

# Get script directory và add TDDet/codes vào path
SCRIPT_DIR = Path(__file__).parent.absolute()
TDDet_CODES_DIR = SCRIPT_DIR / "codes"
# Add codes directory vào sys.path để có thể import như package
sys.path.insert(0, str(TDDet_CODES_DIR))

# CRITICAL: Import ultralytics TRƯỚC KHI import TDDet modules
# Vì TDDet/codes/nn/tasks.py import từ ultralytics.utils và các modules khác
print("Step 0: Importing ultralytics...")
try:
    import ultralytics
    import ultralytics.utils
    import ultralytics.nn.modules
    # Patch ultralytics.nn.extra_modules nếu chưa có
    import types
    if 'ultralytics.nn.extra_modules' not in sys.modules:
        extra_modules = types.ModuleType('ultralytics.nn.extra_modules')
        sys.modules['ultralytics.nn.extra_modules'] = extra_modules
    
    # Patch yaml_load từ TDDet vào ultralytics.utils nếu chưa có
    if not hasattr(ultralytics.utils, 'yaml_load'):
        # Import yaml_load từ TDDet utils
        from utils import yaml_load
        ultralytics.utils.yaml_load = yaml_load
        print("✓ Patched yaml_load from TDDet into ultralytics.utils")
    
    # Patch make_divisible từ TDDet vào ultralytics.utils.torch_utils nếu chưa có
    import ultralytics.utils.torch_utils as torch_utils_module
    if not hasattr(torch_utils_module, 'make_divisible'):
        # Import make_divisible từ TDDet utils.torch_utils
        from utils.torch_utils import make_divisible
        torch_utils_module.make_divisible = make_divisible
        print("✓ Patched make_divisible from TDDet into ultralytics.utils.torch_utils")

    # Patch TORCH_1_10 flag nếu ultralytics bản mới không còn export
    try:
        import ultralytics.utils.tal as tal
        if not hasattr(tal, "TORCH_1_10"):
            # Đặt False (mặc định coi như dùng torch >= 1.11, nhánh code mới hơn)
            tal.TORCH_1_10 = False
            print("✓ Patched TORCH_1_10 flag into ultralytics.utils.tal")
    except Exception as _e:
        # Không critical, chỉ log cảnh báo
        print(f"⚠ Warning: could not patch TORCH_1_10 in ultralytics.utils.tal: {_e}")
    
    # Đảm bảo ultralytics.nn.backbone được tạo và set như attribute
    import types
    if 'ultralytics.nn.backbone' not in sys.modules:
        backbone_module = types.ModuleType('ultralytics.nn.backbone')
        backbone_module.__path__ = []
        sys.modules['ultralytics.nn.backbone'] = backbone_module
        # Set như attribute của ultralytics.nn
        ultralytics.nn.backbone = backbone_module
    
    print("✓ Ultralytics imported successfully")
except Exception as e:
    print(f"⚠ Warning: Could not import ultralytics: {e}")
    # Nếu không import được, tạo fake modules
    import types
    if 'ultralytics' not in sys.modules:
        ultralytics_module = types.ModuleType('ultralytics')
        ultralytics_module.__path__ = []
        sys.modules['ultralytics'] = ultralytics_module
    if 'ultralytics.utils' not in sys.modules:
        sys.modules['ultralytics.utils'] = types.ModuleType('ultralytics.utils')
    if 'ultralytics.nn' not in sys.modules:
        nn_module = types.ModuleType('ultralytics.nn')
        nn_module.__path__ = []
        sys.modules['ultralytics.nn'] = nn_module
    if 'ultralytics.nn.modules' not in sys.modules:
        sys.modules['ultralytics.nn.modules'] = types.ModuleType('ultralytics.nn.modules')
    if 'ultralytics.nn.extra_modules' not in sys.modules:
        sys.modules['ultralytics.nn.extra_modules'] = types.ModuleType('ultralytics.nn.extra_modules')

# CRITICAL: Patch ultralytics.nn.backbone TRƯỚC KHI import TDDet modules
# Vì tasks.py import từ ultralytics.nn.backbone.* ngay khi được import
print("Step 0.5: Patching ultralytics.nn.backbone...")
import types
# Đảm bảo ultralytics.nn.backbone được tạo và đăng ký
if 'ultralytics.nn.backbone' not in sys.modules:
    backbone_module = types.ModuleType('ultralytics.nn.backbone')
    backbone_module.__path__ = []
    sys.modules['ultralytics.nn.backbone'] = backbone_module
    # Đảm bảo ultralytics.nn có attribute backbone
    if not hasattr(ultralytics.nn, 'backbone'):
        ultralytics.nn.backbone = backbone_module

# CRITICAL: Patch backbone modules vào ultralytics.nn.backbone TRƯỚC KHI import TDDet modules
# Vì tasks.py import từ ultralytics.nn.backbone.* ngay khi được import
print("Step 0.5: Patching backbone modules...")
try:
    # Import các backbone modules từ TDDet và patch vào ultralytics.nn.backbone
    # Import từng module một cách cẩn thận để tránh trigger import của tasks.py
    import importlib.util
    
    backbone_files = [
        'convnextv2', 'fasternet', 'efficientViT', 'EfficientFormerV2', 'VanillaNet',
        'revcol', 'lsknet', 'SwinTransformer', 'repvit', 'CSwomTramsformer',
        'UniRepLKNet', 'TransNext', 'rmt', 'pkinet', 'mobilenetv4'
    ]
    
    # Đảm bảo ultralytics.nn.backbone tồn tại và có thể truy cập
    if 'ultralytics.nn.backbone' not in sys.modules:
        backbone_module = types.ModuleType('ultralytics.nn.backbone')
        backbone_module.__path__ = []
        sys.modules['ultralytics.nn.backbone'] = backbone_module
    if not hasattr(ultralytics.nn, 'backbone'):
        ultralytics.nn.backbone = sys.modules['ultralytics.nn.backbone']
    
    # Chuẩn bị package alias cho TransNeXt (thư mục con) để tránh lỗi import lẫn lộn chữ hoa/thường
    transnext_pkg_name = "ultralytics.nn.backbone.TransNeXt"
    if transnext_pkg_name not in sys.modules:
        import types as _types
        transnext_pkg = _types.ModuleType(transnext_pkg_name)
        transnext_pkg.__path__ = [str(TDDet_CODES_DIR / "nn" / "backbone" / "TransNeXt")]
        sys.modules[transnext_pkg_name] = transnext_pkg
        # cũng set attribute trên ultralytics.nn.backbone nếu cần
        setattr(ultralytics.nn.backbone, "TransNeXt", transnext_pkg)

    for module_name in backbone_files:
        try:
            module_path = TDDet_CODES_DIR / "nn" / "backbone" / f"{module_name}.py"
            if module_path.exists():
                # Đăng ký với tên đầy đủ trong sys.modules
                full_module_name = f"ultralytics.nn.backbone.{module_name}"
                spec = importlib.util.spec_from_file_location(full_module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # Đăng ký vào sys.modules với tên đầy đủ
                sys.modules[full_module_name] = module
                # Cũng set attribute để có thể truy cập
                setattr(ultralytics.nn.backbone, module_name, module)
                # Alias cho TransNeXt (khác biệt chữ hoa) nếu có
                if module_name.lower() == "transnext":
                    sys.modules["ultralytics.nn.backbone.TransNeXt"] = module
                    # cũng gắn vào package đã tạo ở trên
                    setattr(sys.modules["ultralytics.nn.backbone.TransNeXt"], "__dict__", module.__dict__)
        except Exception as e:
            # Một số modules có thể không import được, bỏ qua
            print(f"⚠ Warning: Could not patch {module_name}: {e}")
    
    print("✓ Patched backbone modules from TDDet into ultralytics.nn.backbone")
except Exception as e:
    print(f"⚠ Warning: Could not patch all backbone modules: {e}")
    import traceback
    traceback.print_exc()

# CRITICAL: Import TDDet modules SAU KHI ultralytics đã được import và backbone đã được patch
# Điều này đảm bảo modules được register trước khi parse_model được gọi
print("Step 1: Importing TDDet modules...")
try:
    
    # Import backbone modules để MobileNetV4ConvLarge có sẵn
    # Import như package để relative imports hoạt động đúng
    from nn.backbone.mobilenetv4 import (
        MobileNetV4ConvSmall,
        MobileNetV4ConvMedium,
        MobileNetV4ConvLarge,
        MobileNetV4HybridMedium,
        MobileNetV4HybridLarge
    )
    print("✓ MobileNetV4ConvLarge imported successfully")
    
    # Import extra_modules để CFF, DySample có sẵn
    from nn.extra_modules.block import CFF, DySample
    print("✓ CFF, DySample imported successfully")
    
    # Import tasks module và patch globals để modules có sẵn trong parse_model
    import ultralytics.nn.tasks as tasks_module
    import builtins
    
    # Register trong tasks module namespace
    tasks_module.MobileNetV4ConvLarge = MobileNetV4ConvLarge
    tasks_module.MobileNetV4ConvSmall = MobileNetV4ConvSmall
    tasks_module.MobileNetV4ConvMedium = MobileNetV4ConvMedium
    tasks_module.MobileNetV4HybridMedium = MobileNetV4HybridMedium
    tasks_module.MobileNetV4HybridLarge = MobileNetV4HybridLarge
    tasks_module.CFF = CFF
    tasks_module.DySample = DySample
    
    # Register trong builtins (fallback)
    builtins.MobileNetV4ConvLarge = MobileNetV4ConvLarge
    builtins.CFF = CFF
    builtins.DySample = DySample
    
    # CRITICAL: Patch parse_model's globals
    func_globals = None
    if hasattr(tasks_module, 'parse_model'):
        parse_model_func = tasks_module.parse_model
        func_globals = parse_model_func.__globals__
        func_globals['MobileNetV4ConvLarge'] = MobileNetV4ConvLarge
        func_globals['MobileNetV4ConvSmall'] = MobileNetV4ConvSmall
        func_globals['MobileNetV4ConvMedium'] = MobileNetV4ConvMedium
        func_globals['MobileNetV4HybridMedium'] = MobileNetV4HybridMedium
        func_globals['MobileNetV4HybridLarge'] = MobileNetV4HybridLarge
        func_globals['CFF'] = CFF
        func_globals['DySample'] = DySample
        print("✓ Patched parse_model.__globals__ with TDDet modules")
        
        # Verify registration
        if 'MobileNetV4ConvLarge' in func_globals:
            print("✓ MobileNetV4ConvLarge found in parse_model.__globals__")
        else:
            print("⚠ Warning: MobileNetV4ConvLarge not found in parse_model.__globals__")
    else:
        print("⚠ Warning: parse_model function not found in tasks_module")
        
except Exception as e:
    print(f"✗ Failed to import TDDet modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Bây giờ import ultralytics entrypoint và chạy
# Phải SAU KHI modules đã được register
print("Step 2: Importing YOLO entrypoint...")
from ultralytics.cfg import entrypoint
print("✓ YOLO entrypoint imported")
print("")

if __name__ == "__main__":
    # Pass tất cả arguments đến yolo CLI
    sys.exit(entrypoint())