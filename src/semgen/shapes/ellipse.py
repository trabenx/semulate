import numpy as np
import cv2
from typing import Tuple, Dict, Any

from .base_shape import BaseShape

class Ellipse(BaseShape):
    """Generates and draws a potentially rotated ellipse."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config (Dict[str, Any]): Must include 'center', 'intensity', 'axes' (tuple (width, height)),
                                     'rotation' (ellipse angle).
                                     Optional: 'border_width'.
                                     Note: 'rotation' in config becomes the ellipse angle.
        """
        super().__init__(config) # Handles center, intensity, base rotation (though ellipse uses its own angle directly)
        axes = config.get('axes', (10, 5)) # Default axes if not provided
        self.axis_width: float = float(axes[0])
        self.axis_height: float = float(axes[1])
        # For cv2.ellipse, axes are half-lengths
        self.half_axis_width = int(round(self.axis_width / 2.0))
        self.half_axis_height = int(round(self.axis_height / 2.0))
        # Ellipse angle is the rotation from config
        self.angle: float = self.rotation
        self.border_width: int = int(config.get('border_width', -1)) # Default filled

    def draw(self, image_data: np.ndarray) -> np.ndarray:
        """Draws the ellipse on the image."""
        draw_intensity, line_type = self._get_draw_params(image_data.dtype)
        center_int = (int(round(self.center[0])), int(round(self.center[1])))
        axes_int = (self.half_axis_width, self.half_axis_height)

        cv2.ellipse(
            img=image_data,
            center=center_int,
            axes=axes_int,
            angle=self.angle,
            startAngle=0,         # Draw full ellipse
            endAngle=360,
            color=draw_intensity,
            thickness=self.border_width,
            lineType=line_type
        )
        return image_data

    def generate_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generates a binary mask for the ellipse."""
        mask = np.zeros(shape, dtype=np.uint8)
        mask_intensity, thickness, line_type = self._get_mask_params()
        center_int = (int(round(self.center[0])), int(round(self.center[1])))
        axes_int = (self.half_axis_width, self.half_axis_height)

        cv2.ellipse(
            img=mask,
            center=center_int,
            axes=axes_int,
            angle=self.angle,
            startAngle=0,
            endAngle=360,
            color=mask_intensity,
            thickness=thickness, # Always filled
            lineType=line_type   # No AA for mask
        )
        return mask

    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """Calculates bounding box of a rotated ellipse (more complex)."""
        # Get points on the ellipse and find their min/max
        # Or use cv2.ellipse2Poly and find min/max of points (approimation)
        center_int = (int(round(self.center[0])), int(round(self.center[1])))
        axes_int = (self.half_axis_width, self.half_axis_height)
        try:
            # Note: ellipse2Poly might not be in all OpenCV versions or require specific args
            # Using a simpler approximation based on rotated rectangle bounding box
            # This is inaccurate but provides a basic bounding box
            angle_rad = np.radians(self.angle)
            cos_a, sin_a = np.abs(np.cos(angle_rad)), np.abs(np.sin(angle_rad))
            bound_w = self.axis_width * cos_a + self.axis_height * sin_a
            bound_h = self.axis_width * sin_a + self.axis_height * cos_a
            cx, cy = self.center
            return (
                int(cx - bound_w / 2), int(cy - bound_h / 2),
                int(cx + bound_w / 2), int(cy + bound_h / 2)
            )
        except Exception: # Fallback if ellipse2Poly fails or isn't available
             cx, cy = self.center
             r_max = max(self.half_axis_width, self.half_axis_height)
             return (int(cx - r_max), int(cy - r_max), int(cx + r_max), int(cy + r_max))