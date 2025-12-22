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

# CRITICAL: Import TDDet modules TRƯỚC KHI import bất cứ gì từ ultralytics
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

