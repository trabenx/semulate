import numpy as np
import cv2
from typing import Tuple, Dict, Any

from .base_shape import BaseShape

class Circle(BaseShape):
    """Generates and draws a circle."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config (Dict[str, Any]): Must include 'center', 'intensity', 'radius'.
                                     Optional: 'rotation' (ignored), 'border_width'.
        """
        super().__init__(config)
        self.radius: float = float(config['radius'])
        # border_width > 0 for outline, <= 0 for filled (using thickness convention)
        self.border_width: int = int(config.get('border_width', -1)) # Default filled


    def draw(self, image_data: np.ndarray) -> np.ndarray:
        """Draws the circle on the image."""
        draw_intensity, line_type = self._get_draw_params(image_data.dtype)
        center_int = (int(round(self.center[0])), int(round(self.center[1])))
        radius_int = int(round(self.radius))

        cv2.circle(
            img=image_data,
            center=center_int,
            radius=radius_int,
            color=draw_intensity,
            thickness=self.border_width, # Use border_width as thickness
            lineType=line_type
        )
        return image_data

    def generate_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generates a binary mask for the circle."""
        mask = np.zeros(shape, dtype=np.uint8)
        mask_intensity, thickness, line_type = self._get_mask_params()
        center_int = (int(round(self.center[0])), int(round(self.center[1])))
        radius_int = int(round(self.radius))

        cv2.circle(
            img=mask,
            center=center_int,
            radius=radius_int,
            color=mask_intensity,
            thickness=thickness, # Always filled for mask
            lineType=line_type
        )
        return mask

    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        cx, cy = self.center
        r = self.radius
        return (int(cx - r), int(cy - r), int(cx + r), int(cy + r))