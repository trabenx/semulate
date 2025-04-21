import numpy as np
import cv2
from typing import Tuple, Dict, Any

from .base_shape import BaseShape

class Line(BaseShape):
    """Generates and draws a straight line segment defined by center, length, angle."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config (Dict[str, Any]): Must include 'center' (tuple), 'intensity',
                                     'length', 'rotation' (line angle), 'thickness'.
                                     Optional: 'amplitude', 'frequency', 'phase' (for wavy - NOT IMPLEMENTED YET).
        """
        # Note: BaseShape expects 'center' and 'rotation'
        if 'center' not in config:
             raise ValueError("Line config missing 'center'.") # Ensure center is calculated beforehand if using %
        if 'rotation' not in config: # Line angle
             config['rotation'] = 0.0 # Default angle if missing
             print("Warning: Line config missing 'rotation', defaulting to 0 degrees.")

        super().__init__(config) # Initializes center, intensity, rotation (which is line angle)

        self.length: float = float(config['length'])
        self.thickness: int = int(config['thickness'])

        # Calculate start and end points based on center, length, angle (rotation)
        cx, cy = self.center
        angle_rad = np.radians(self.rotation) # Use the shape's rotation as the line angle
        half_len = self.length / 2.0

        dx = half_len * np.cos(angle_rad)
        dy = half_len * np.sin(angle_rad) # Standard angle math

        self.start_point = (cx - dx, cy - dy)
        self.end_point = (cx + dx, cy + dy)

        self.is_wavy = False # Placeholder


    def _get_line_points(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get integer start/end points."""
        # Start/end points are already calculated in __init__
        pt1 = (int(round(self.start_point[0])), int(round(self.start_point[1])))
        pt2 = (int(round(self.end_point[0])), int(round(self.end_point[1])))
        return pt1, pt2

    def draw(self, image_data: np.ndarray) -> np.ndarray:
        """Draws the line segment on the image."""
        if self.is_wavy:
            # TODO: Implement wavy line point generation and cv2.polylines
            print("Warning: Wavy line drawing not implemented.")
            return image_data

        pt1, pt2 = self._get_line_points()
        draw_intensity, line_type = self._get_draw_params(image_data.dtype)

        cv2.line(
            img=image_data,
            pt1=pt1,
            pt2=pt2,
            color=draw_intensity,
            thickness=self.thickness,
            lineType=line_type
        )
        return image_data

    def generate_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generates a binary mask for the line segment."""
        mask = np.zeros(shape, dtype=np.uint8)
        if self.is_wavy:
            # TODO: Implement wavy line mask generation
            print("Warning: Wavy line mask generation not implemented.")
            return mask

        pt1, pt2 = self._get_line_points()
        mask_intensity, _, line_type = self._get_mask_params() # Non-AA

        cv2.line(
            img=mask,
            pt1=pt1,
            pt2=pt2,
            color=mask_intensity,
            thickness=self.thickness,
            lineType=line_type # Use LINE_8 for binary mask
        )
        return mask

    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        pt1, pt2 = self._get_line_points()
        xmin = min(pt1[0], pt2[0]) - self.thickness // 2
        ymin = min(pt1[1], pt2[1]) - self.thickness // 2
        xmax = max(pt1[0], pt2[0]) + self.thickness // 2
        ymax = max(pt1[1], pt2[1]) + self.thickness // 2
        return (int(xmin), int(ymin), int(xmax), int(ymax))