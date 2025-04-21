# Expose the main generation function and config loader
from .pipeline import generate_sample
from .config_loader import load_config
from .config_randomizer import randomize_config_for_sample # Added

# Optionally expose other core components if needed by external tools
from .core import GeneratedSample
from .raffle import Raffler
# Expose factories?
from .shapes import create_shape
from .artifacts import create_artifact
from .noise import create_noise

__version__ = "0.1.0" # Basic versioning

__all__ = [
    "generate_sample",
    "load_config",
    "randomize_config_for_sample", # Added
    "GeneratedSample",
    "Raffler",
    "create_shape",
    "create_artifact",
    "create_noise",
    "__version__",
]