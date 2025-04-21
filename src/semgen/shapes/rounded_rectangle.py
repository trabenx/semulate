import numpy as np
import cv2
from typing import Tuple, Dict, Any

from .base_shape import BaseShape

class RoundedRectangle(BaseShape):
    """
    Generates and draws a potentially rotated rectangle with rounded corners,
    approximated using polygon segments.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config (Dict[str, Any]): Must include 'center', 'intensity', 'width', 'height',
                                     'corner_radius'.
                                     Optional: 'rotation', 'border_width'.
        """
        super().__init__(config)
        self.width: float = float(config['width'])
        self.height: float = float(config['height'])
        raw_radius = float(config['corner_radius'])
        # Ensure radius isn't too large
        self.corner_radius = min(raw_radius, self.width / 2.0, self.height / 2.0)
        self.border_width: int = int(config.get('border_width', -1)) # Default filled
        self.corner_segments = 10 # Number of line segments to approximate each 90-degree corner

    def _generate_rounded_vertices(self) -> np.ndarray:
        """Generates vertices for the rounded rectangle polygon approximation."""
        w = self.width
        h = self.height
        r = self.corner_radius

        if r <= 0: # Just a normal rectangle
             hw, hh = w / 2, h / 2
             return np.array([[-hw,-hh], [hw,-hh], [hw,hh], [-hw,hh]], dtype=np.float32)

        vertices = []

        # Calculate centers of the four corner arcs
        arc_centers = [
            (w/2 - r, h/2 - r),  # Top-right
            (-w/2 + r, h/2 - r), # Top-left
            (-w/2 + r, -h/2 + r),# Bottom-left
            (w/2 - r, -h/2 + r) # Bottom-right
        ]

        # Generate points for each corner arc and straight segment
        start_angles = [0, 90, 180, 270] # Degrees

        for i in range(4):
            cx, cy = arc_centers[i]
            start_angle = start_angles[i]
            # Add vertices for the arc
            for j in range(self.corner_segments + 1):
                angle = np.radians(start_angle + (j / self.corner_segments) * 90.0)
                vx = cx + r * np.cos(angle)
                vy = cy + r * np.sin(angle)
                vertices.append([vx, vy])

        # Vertices are generated centered at origin
        return np.array(vertices, dtype=np.float32)


    def _get_transformed_vertices(self) -> np.ndarray:
        """Transforms the rounded vertices (scale is implicit, rotate, translate)."""
        base_vertices = self._generate_rounded_vertices()
        cx, cy = self.center
        angle_rad = np.radians(self.rotation)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Rotate vertices around origin
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_vertices = base_vertices @ rotation_matrix.T

        # Translate to center
        transformed_vertices = rotated_vertices + np.array([cx, cy])

        return transformed_vertices.astype(np.int32)


    def draw(self, image_data: np.ndarray) -> np.ndarray:
        """Draws the rounded rectangle on the image."""
        vertices = self._get_transformed_vertices()
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
                lineType=line_type # Use AA fill if possible/desired
            )
        return image_data

    def generate_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generates a binary mask for the rounded rectangle."""
        mask = np.zeros(shape, dtype=np.uint8)
        vertices = self._get_transformed_vertices()
        mask_intensity, _, line_type = self._get_mask_params() # Use non-AA for mask

        cv2.fillPoly(
            img=mask,
            pts=[vertices],
            color=mask_intensity,
            lineType=line_type # Use LINE_8 for binary mask
        )
        return mask

    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        # Bounding box of the transformed vertices
        vertices = self._get_transformed_vertices()
        xmin = np.min(vertices[:, 0])
        ymin = np.min(vertices[:, 1])
        xmax = np.max(vertices[:, 0])
        ymax = np.max(vertices[:, 1])
        return (int(xmin), int(ymin), int(xmax), int(ymax))