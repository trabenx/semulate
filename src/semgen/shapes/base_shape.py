import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class BaseShape(ABC):
    """
    Abstract base class for all shapes that can be drawn on the SEM image.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the shape with its configuration.

        Args:
            config (Dict[str, Any]): Dictionary containing shape parameters like
                                     center, intensity, rotation, and shape-specific
                                     parameters (e.g., radius, width, height).
                                     Must include 'center' (tuple), 'intensity' (float),
                                     'rotation' (float degrees).
        """
        self.config = config
        self.center: Tuple[float, float] = tuple(map(float, config['center']))
        self.intensity: float = float(config['intensity'])
        # Rotation in degrees, counter-clockwise is positive in OpenCV
        self.rotation: float = float(config.get('rotation', 0.0))
        # Assume anti-aliasing unless specified otherwise in global/layer config
        self.anti_aliasing: bool = config.get('anti_aliasing', True)


    @abstractmethod
    def draw(self, image_data: np.ndarray) -> np.ndarray:
        """
        Draw the shape onto the provided image data.

        Args:
            image_data (np.ndarray): The numpy array representing the image canvas.
                                     Expected to be float32 [0,1] or uint8/uint16.

        Returns:
            np.ndarray: The image data with the shape drawn on it.
        """
        pass

    @abstractmethod
    def generate_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Generate a binary mask (uint8) of the shape.

        Args:
            shape (Tuple[int, int]): The desired (height, width) of the mask array.

        Returns:
            np.ndarray: A binary (0 or 1) mask of dtype uint8.
        """
        pass

    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """
        Calculate the axis-aligned bounding box of the shape after rotation.
        (Optional but potentially useful for optimizations or local artifacts)

        Returns:
            Tuple[int, int, int, int]: (xmin, ymin, xmax, ymax)
        """
        # Default implementation - subclasses should override for accuracy
        # This is a rough estimate based on center only
        size_estimate = 10 # Default guess - very inaccurate
        cx, cy = self.center
        return (
            int(cx - size_estimate // 2),
            int(cy - size_estimate // 2),
            int(cx + size_estimate // 2),
            int(cy + size_estimate // 2)
        )

    def _get_draw_params(self, image_dtype: np.dtype):
        """Helper to get drawing color and line type based on image dtype."""
        if np.issubdtype(image_dtype, np.floating):
            draw_intensity = self.intensity
        elif np.issubdtype(image_dtype, np.integer):
            max_val = np.iinfo(image_dtype).max
            draw_intensity = int(self.intensity * max_val)
        else:
            # Default for uint8 if type unknown, adjust as needed
            draw_intensity = int(self.intensity * 255)

        line_type = cv2.LINE_AA if self.anti_aliasing else cv2.LINE_8
        return draw_intensity, line_type

    def _get_mask_params(self):
        """Helper to get parameters for drawing masks (filled, no AA)."""
        mask_intensity = 1 # Binary mask value
        thickness = -1 # Fill shape
        line_type = cv2.LINE_8 # No anti-aliasing for crisp binary mask edges
        return mask_intensity, thickness, line_type