from typing import Dict, Any, Type

from .base_shape import BaseShape
from .circle import Circle
from .rectangle import Rectangle
from .custom_polygon import CustomPolygon
from .line import Line
from .ellipse import Ellipse
from .rounded_rectangle import RoundedRectangle
from .worm import Worm

# Import other shape classes as they are created (Ellipse, RegularPolygon, etc.)

# Map shape type names (lowercase) to their corresponding classes
SHAPE_CLASS_MAP: Dict[str, Type[BaseShape]] = {
    "circle": Circle,
    "rectangle": Rectangle,
    "ellipse": Ellipse,                   # Added
    "rounded_rectangle": RoundedRectangle,# Added
    "h": CustomPolygon,
    "l": CustomPolygon,
    "e": CustomPolygon,
    "s": CustomPolygon,
    "t": CustomPolygon,
    "u": CustomPolygon,
    "x": CustomPolygon,
    "y": CustomPolygon,
    "i": CustomPolygon,
    "c": CustomPolygon,
    "line": Line,
    "worm": Worm,                         # Added
    # Add other shapes here:
    # "polygon": RegularPolygon,
}

def create_shape(shape_config: Dict[str, Any]) -> BaseShape:
    """
    Factory function to create a shape instance based on configuration.

    Args:
        shape_config (Dict[str, Any]): Configuration dictionary for the shape.
                                       Must include 'shape_type' key corresponding
                                       to a key in SHAPE_CLASS_MAP. For custom
                                       polygons ('h', 'l', etc.), 'shape_name' is
                                       automatically added/used internally by CustomPolygon.

    Returns:
        BaseShape: An instance of the requested shape class.

    Raises:
        ValueError: If the shape_type is unknown or not supported.
    """
    shape_type = shape_config.get('shape_type', '').lower()

    if not shape_type:
        raise ValueError("Shape configuration must include a 'shape_type' key.")

    ShapeClass = SHAPE_CLASS_MAP.get(shape_type)

    if ShapeClass is None:
        raise ValueError(f"Unknown or unsupported shape_type: '{shape_type}'")

    # For custom polygons, ensure shape_name is passed if needed
    if ShapeClass == CustomPolygon and 'shape_name' not in shape_config:
        shape_config['shape_name'] = shape_type.upper() # Use the type key as name

    try:
        return ShapeClass(config=shape_config)
    except KeyError as e:
        raise ValueError(f"Missing required parameter '{e}' for shape_type '{shape_type}'") from e
    except Exception as e:
        raise RuntimeError(f"Error instantiating shape_type '{shape_type}': {e}") from e