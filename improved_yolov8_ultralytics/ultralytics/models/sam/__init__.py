# Ultralytics YOLO 🚀, AGPL-3.0 license

# SAM modules may not be available if modules directory is missing
try:
    from .model import SAM
    from .predict import Predictor
    __all__ = "SAM", "Predictor"  # tuple or list
except (ImportError, ModuleNotFoundError):
    SAM = None
    Predictor = None
    __all__ = ()  # empty tuple
