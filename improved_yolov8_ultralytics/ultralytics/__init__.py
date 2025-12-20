# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.1.9"

from ultralytics.data.explorer.explorer import Explorer
from ultralytics.models import RTDETR, YOLO

# Optional imports
try:
    from ultralytics.models import SAM
except (ImportError, ModuleNotFoundError):
    SAM = None

try:
from ultralytics.models.fastsam import FastSAM
except (ImportError, ModuleNotFoundError):
    FastSAM = None

try:
from ultralytics.models.nas import NAS
except (ImportError, ModuleNotFoundError):
    NAS = None
from ultralytics.utils import ASSETS, SETTINGS as settings
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)
