# Expose the factory function and base class for easier import
from .base_shape import BaseShape
from .shape_factory import create_shape

# Optionally expose individual shape classes if needed elsewhere
from .circle import Circle
from .rectangle import Rectangle
from .ellipse import Ellipse # Added
from .rounded_rectangle import RoundedRectangle # Added
from .custom_polygon import CustomPolygon
from .line import Line
from .worm import Worm # Added

__all__ = [
    "BaseShape",
    "create_shape",
    "Circle",
    "Rectangle",
    "Ellipse", # Added
    "RoundedRectangle", # Added
    "CustomPolygon",
    "Line",
    "Worm", # Added
]