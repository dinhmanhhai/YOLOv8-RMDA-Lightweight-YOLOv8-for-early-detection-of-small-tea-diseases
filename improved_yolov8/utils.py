"""
Utility functions for registering custom modules with ultralytics.
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import custom modules
from models.blocks import MixSPPF, RFCBAM_Neck, C2f_RFCBAM
from models.repgfpn import RepGFPN
from models.rfcbam import RFCBAMConv
from models.akconv import AKConv
from models.dyhead import Detect_DyHead, DyHeadBlockLite

# Register modules with ultralytics
def register_custom_modules():
    """
    Register custom modules with ultralytics.nn.modules and ultralytics.nn.tasks.
    This allows YAML config files to use these modules.
    """
    import ultralytics.nn.modules as ultralytics_modules
    
    # Register MixSPPF
    if not hasattr(ultralytics_modules, 'MixSPPF'):
        ultralytics_modules.MixSPPF = MixSPPF
        if hasattr(ultralytics_modules, '__all__'):
            ultralytics_modules.__all__ = list(ultralytics_modules.__all__) + ['MixSPPF']
    
    # Register RFCBAMConv
    if not hasattr(ultralytics_modules, 'RFCBAMConv'):
        ultralytics_modules.RFCBAMConv = RFCBAMConv
        if hasattr(ultralytics_modules, '__all__'):
            ultralytics_modules.__all__ = list(ultralytics_modules.__all__) + ['RFCBAMConv']
    
    # Register RFCBAM_Neck
    if not hasattr(ultralytics_modules, 'RFCBAM_Neck'):
        ultralytics_modules.RFCBAM_Neck = RFCBAM_Neck
        if hasattr(ultralytics_modules, '__all__'):
            ultralytics_modules.__all__ = list(ultralytics_modules.__all__) + ['RFCBAM_Neck']
    
    # Register C2f_RFCBAM
    if not hasattr(ultralytics_modules, 'C2f_RFCBAM'):
        ultralytics_modules.C2f_RFCBAM = C2f_RFCBAM
        if hasattr(ultralytics_modules, '__all__'):
            ultralytics_modules.__all__ = list(ultralytics_modules.__all__) + ['C2f_RFCBAM']
    
    # Register RepGFPN
    if not hasattr(ultralytics_modules, 'RepGFPN'):
        ultralytics_modules.RepGFPN = RepGFPN
        if hasattr(ultralytics_modules, '__all__'):
            ultralytics_modules.__all__ = list(ultralytics_modules.__all__) + ['RepGFPN']

    # Register DyHead
    if not hasattr(ultralytics_modules, 'Detect_DyHead'):
        ultralytics_modules.Detect_DyHead = Detect_DyHead
        if hasattr(ultralytics_modules, '__all__'):
            ultralytics_modules.__all__ = list(ultralytics_modules.__all__) + ['Detect_DyHead']
    if not hasattr(ultralytics_modules, 'DyHeadBlockLite'):
        ultralytics_modules.DyHeadBlockLite = DyHeadBlockLite
        if hasattr(ultralytics_modules, '__all__'):
            ultralytics_modules.__all__ = list(ultralytics_modules.__all__) + ['DyHeadBlockLite']
    
    # Register AKConv
    if not hasattr(ultralytics_modules, 'AKConv'):
        ultralytics_modules.AKConv = AKConv
        if hasattr(ultralytics_modules, '__all__'):
            ultralytics_modules.__all__ = list(ultralytics_modules.__all__) + ['AKConv']
    
    # Also register in extra_modules if it exists
    try:
        import ultralytics.nn.extra_modules as extra_modules
        if not hasattr(extra_modules, 'MixSPPF'):
            extra_modules.MixSPPF = MixSPPF
        if not hasattr(extra_modules, 'RFCBAMConv'):
            extra_modules.RFCBAMConv = RFCBAMConv
        if not hasattr(extra_modules, 'RFCBAM_Neck'):
            extra_modules.RFCBAM_Neck = RFCBAM_Neck
        if not hasattr(extra_modules, 'C2f_RFCBAM'):
            extra_modules.C2f_RFCBAM = C2f_RFCBAM
        if not hasattr(extra_modules, 'RepGFPN'):
            extra_modules.RepGFPN = RepGFPN
        if not hasattr(extra_modules, 'AKConv'):
            extra_modules.AKConv = AKConv
        if not hasattr(extra_modules, 'Detect_DyHead'):
            extra_modules.Detect_DyHead = Detect_DyHead
        if not hasattr(extra_modules, 'DyHeadBlockLite'):
            extra_modules.DyHeadBlockLite = DyHeadBlockLite
    except ImportError:
        pass
    
    # CRITICAL: Register in ultralytics.nn.tasks globals for parse_model
    # parse_model uses globals() to find modules, so we need to register in the right namespace
    try:
        import ultralytics.nn.tasks as tasks_module
        import builtins
        
        # Register in tasks module namespace (where parse_model is defined)
        tasks_module.RFCBAMConv = RFCBAMConv
        tasks_module.MixSPPF = MixSPPF
        tasks_module.RFCBAM_Neck = RFCBAM_Neck
        tasks_module.C2f_RFCBAM = C2f_RFCBAM
        tasks_module.RepGFPN = RepGFPN
        tasks_module.AKConv = AKConv
        tasks_module.Detect_DyHead = Detect_DyHead
        tasks_module.DyHeadBlockLite = DyHeadBlockLite
        
        # Also register in builtins for global lookup (fallback)
        builtins.RFCBAMConv = RFCBAMConv
        builtins.MixSPPF = MixSPPF
        builtins.RFCBAM_Neck = RFCBAM_Neck
        builtins.C2f_RFCBAM = C2f_RFCBAM
        builtins.RepGFPN = RepGFPN
        builtins.AKConv = AKConv
        builtins.Detect_DyHead = Detect_DyHead
        builtins.DyHeadBlockLite = DyHeadBlockLite
        
        # CRITICAL: Patch parse_model's globals by monkey-patching
        # This ensures modules are available when parse_model calls globals()
        if hasattr(tasks_module, 'parse_model'):
            parse_model_func = tasks_module.parse_model
            try:
                # Modify __globals__ directly - this is the key!
                func_globals = parse_model_func.__globals__
                func_globals['RFCBAMConv'] = RFCBAMConv
                func_globals['MixSPPF'] = MixSPPF
                func_globals['RFCBAM_Neck'] = RFCBAM_Neck
                func_globals['C2f_RFCBAM'] = C2f_RFCBAM
                func_globals['RepGFPN'] = RepGFPN
                func_globals['AKConv'] = AKConv
                func_globals['Detect_DyHead'] = Detect_DyHead
                func_globals['DyHeadBlockLite'] = DyHeadBlockLite
                print("✓ Patched parse_model.__globals__ with custom modules")
            except (AttributeError, TypeError) as e:
                print(f"⚠ Could not patch parse_model.__globals__: {e}")
                # Fallback: try to wrap the function
                try:
                    original_parse_model = parse_model_func
                    def wrapped_parse_model(d, ch, verbose=True):
                        # Add modules to current globals before calling
                        import sys
                        frame = sys._getframe(1)
                        frame.f_globals['RFCBAMConv'] = RFCBAMConv
                        frame.f_globals['MixSPPF'] = MixSPPF
                        frame.f_globals['RFCBAM_Neck'] = RFCBAM_Neck
                        frame.f_globals['C2f_RFCBAM'] = C2f_RFCBAM
                        frame.f_globals['RepGFPN'] = RepGFPN
                        frame.f_globals['AKConv'] = AKConv
                        frame.f_globals['Detect_DyHead'] = Detect_DyHead
                        frame.f_globals['DyHeadBlockLite'] = DyHeadBlockLite
                        return original_parse_model(d, ch, verbose=verbose)
                    tasks_module.parse_model = wrapped_parse_model
                except Exception:
                    pass
        
    except Exception as e:
        print(f"Warning: Could not register modules in tasks module: {e}")
        import traceback
        traceback.print_exc()


# Auto-register on import
register_custom_modules()

