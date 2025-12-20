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
    Register custom modules with ultralytics.nn.modules.
    This allows YAML config files to use these modules.
    """
    import ultralytics.nn.modules as ultralytics_modules
    
    # Register MixSPPF
    if not hasattr(ultralytics_modules, 'MixSPPF'):
        ultralytics_modules.MixSPPF = MixSPPF
        ultralytics_modules.__all__ = list(ultralytics_modules.__all__) + ['MixSPPF']
    
    # Register RFCBAMConv
    if not hasattr(ultralytics_modules, 'RFCBAMConv'):
        ultralytics_modules.RFCBAMConv = RFCBAMConv
        ultralytics_modules.__all__ = list(ultralytics_modules.__all__) + ['RFCBAMConv']
    
    # Register RFCBAM_Neck
    if not hasattr(ultralytics_modules, 'RFCBAM_Neck'):
        ultralytics_modules.RFCBAM_Neck = RFCBAM_Neck
        ultralytics_modules.__all__ = list(ultralytics_modules.__all__) + ['RFCBAM_Neck']
    
    # Register C2f_RFCBAM
    if not hasattr(ultralytics_modules, 'C2f_RFCBAM'):
        ultralytics_modules.C2f_RFCBAM = C2f_RFCBAM
        ultralytics_modules.__all__ = list(ultralytics_modules.__all__) + ['C2f_RFCBAM']
    
    # Register RepGFPN
    if not hasattr(ultralytics_modules, 'RepGFPN'):
        ultralytics_modules.RepGFPN = RepGFPN
        ultralytics_modules.__all__ = list(ultralytics_modules.__all__) + ['RepGFPN']

    # Register DyHead
    if not hasattr(ultralytics_modules, 'Detect_DyHead'):
        ultralytics_modules.Detect_DyHead = Detect_DyHead
        ultralytics_modules.__all__ = list(ultralytics_modules.__all__) + ['Detect_DyHead']
    if not hasattr(ultralytics_modules, 'DyHeadBlockLite'):
        ultralytics_modules.DyHeadBlockLite = DyHeadBlockLite
        ultralytics_modules.__all__ = list(ultralytics_modules.__all__) + ['DyHeadBlockLite']
    
    # Register AKConv
    if not hasattr(ultralytics_modules, 'AKConv'):
        ultralytics_modules.AKConv = AKConv
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


# Auto-register on import
register_custom_modules()

