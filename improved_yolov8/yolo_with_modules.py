#!/usr/bin/env python3
"""
Wrapper to ensure custom modules are registered before running YOLO CLI.
This script imports modules then executes yolo command.
"""

import sys
import os
from pathlib import Path

# Get script directory and add to path
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

# CRITICAL: Import and register custom modules BEFORE importing ANYTHING from ultralytics
# This ensures modules are registered before parse_model is called
print("Step 1: Registering custom modules...")
try:
    # Import utils which will register all modules
    from improved_yolov8 import utils
    print("✓ Custom modules registered successfully")
    
    # Verify modules are registered
    import ultralytics.nn.tasks as tasks_module
    if hasattr(tasks_module, 'RFCBAMConv'):
        print("✓ RFCBAMConv found in tasks module")
    else:
        print("⚠ Warning: RFCBAMConv not found in tasks module")
        
    # Verify parse_model globals
    if hasattr(tasks_module, 'parse_model'):
        func_globals = tasks_module.parse_model.__globals__
        if 'RFCBAMConv' in func_globals:
            print("✓ RFCBAMConv found in parse_model.__globals__")
        else:
            print("⚠ Warning: RFCBAMConv not found in parse_model.__globals__")
            
except Exception as e:
    print(f"✗ Failed to register custom modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Now import ultralytics entrypoint and run
# This must be AFTER modules are registered
print("Step 2: Importing YOLO entrypoint...")
from ultralytics.cfg import entrypoint
print("✓ YOLO entrypoint imported")
print("")

if __name__ == "__main__":
    # Pass all arguments to yolo CLI
    sys.exit(entrypoint())

