import numpy as np
import cv2
from typing import Tuple, Dict, Any

from .base_shape import BaseShape

class Rectangle(BaseShape):
    """Generates and draws a potentially rotated rectangle."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config (Dict[str, Any]): Must include 'center', 'intensity', 'width', 'height'.
                                     Optional: 'rotation', 'border_width'.
        """
        super().__init__(config)
        self.width: float = float(config['width'])
        self.height: float = float(config['height'])
        self.border_width: int = int(config.get('border_width', -1)) # Default filled

    def _get_rotated_vertices(self) -> np.ndarray:
        """Calculates the 4 vertices of the rotated rectangle."""
        cx, cy = self.center
        width = self.width
        height = self.height
        angle_rad = np.radians(self.rotation)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Center vertices around origin
        hw, hh = width / 2, height / 2
        vertices = np.array([
            [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
        ])

        # Rotate
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_vertices = vertices @ rotation_matrix.T # Apply rotation

        # Translate to center
        rotated_vertices += np.array([cx, cy])

        return rotated_vertices.astype(np.int32) # cv2 needs int32 for polys


    def draw(self, image_data: np.ndarray) -> np.ndarray:
        """Draws the rectangle on the image."""
        vertices = self._get_rotated_vertices()
        draw_intensity, line_type = self._get_draw_params(image_data.dtype)

        if self.border_width > 0: # Draw outline
            cv2.polylines(
                img=image_data,
                pts=[vertices],
                isClosed=True,
                color=draw_intensity,
                thickness=self.border_width,
                lineType=line_type
            )
        else: # Draw filled
            cv2.fillPoly(
                img=image_data,
                pts=[vertices],
                color=draw_intensity,
                lineType=line_type
            )
        return image_data

    def generate_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generates a binary mask for the rectangle."""
        mask = np.zeros(shape, dtype=np.uint8)
        vertices = self._get_rotated_vertices()
        mask_intensity, _, line_type = self._get_mask_params() # Use non-AA for mask

        cv2.fillPoly(
            img=mask,
            pts=[vertices],
            color=mask_intensity,
            lineType=line_type # Use LINE_8 for binary mask
        )
        return mask

    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        vertices = self._get_rotated_vertices()
        xmin = np.min(vertices[:, 0])
        ymin = np.min(vertices[:, 1])
        xmax = np.max(vertices[:, 0])
        ymax = np.max(vertices[:, 1])
        return (int(xmin), int(ymin), int(xmax), int(ymax))

# --- NOTE on Rounded Rectangles ---
# Implementing true rounded rectangles efficiently requires more complex
# polygon generation or drawing arcs. A separate class 'RoundedRectangle'
# would be needed, possibly approximating the corners with polygon segments
# or using a library with dedicated support if performance is critical.
# For now, this standard Rectangle class is provided.