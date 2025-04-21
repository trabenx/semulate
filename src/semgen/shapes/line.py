import numpy as np
import cv2
from typing import Tuple, Dict, Any

from .base_shape import BaseShape

class Line(BaseShape):
    """Generates and draws a straight line segment defined by center, length, angle."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config (Dict[str, Any]): Must include 'center', 'intensity', 'length',
                                     'rotation' (line angle), 'thickness'.
                                     Optional for wavy: 'is_wavy' (bool), 'amplitude',
                                     'frequency', 'phase'.
        """
        # Note: BaseShape expects 'center' and 'rotation'
        if 'center' not in config:
             raise ValueError("Line config missing 'center'.") # Ensure center is calculated beforehand if using %
        if 'rotation' not in config: # Line angle
             config['rotation'] = 0.0 # Default angle if missing
             print("Warning: Line config missing 'rotation', defaulting to 0 degrees.")

        super().__init__(config) # Initializes center, intensity, rotation (which is line angle)

        self.length: float = float(self.get_param('length', 100.0)) # Use get_param now
        self.thickness: int = int(self.get_param('thickness', 2))
        
        # --- Ensure center is usable for calculations ---
        center_arr = np.array(self.center, dtype=np.float64) # Convert center tuple to array

        angle_rad = np.radians(self.rotation)
        half_len = self.length / 2.0

        # Calculate offset vector
        offset_vec = np.array([half_len * np.cos(angle_rad),
                               half_len * np.sin(angle_rad)], dtype=np.float64)

        # --- Define start/end points explicitly as numpy arrays ---
        self.start_point = center_arr - offset_vec
        self.end_point   = center_arr + offset_vec
        # --- Ensure they remain arrays ---

        # Wavy parameters
        self.is_wavy: bool = self.get_param('is_wavy', False)
        self.amplitude: float = float(self.get_param('amplitude', 5.0))
        self.frequency: float = float(self.get_param('frequency', 0.1))
        self.phase: float = float(self.get_param('phase', 0.0))

        self.wavy_path_points = None
        if self.is_wavy and self.length > 0:
            # _generate_wavy_path now receives guaranteed numpy arrays
            self.wavy_path_points = self._generate_wavy_path()

    def _generate_wavy_path(self) -> np.ndarray:
        """Generates points along the wavy line curve."""
        num_segments = max(10, int(self.length / 5)) # More segments for longer/curvier lines
        t = np.linspace(0, 1, num_segments) # Parameter along the line segment

        # Base points along the straight line
        base_points = self.start_point + t[:, np.newaxis] * (self.end_point - self.start_point)

        # Calculate perpendicular direction vector
        direction_vec = self.end_point - self.start_point
        # Normalize direction (handle zero length case)
        norm = np.linalg.norm(direction_vec)
        if norm < 1e-6: return base_points # Not wavy if length is zero

        perp_vec = np.array([-direction_vec[1], direction_vec[0]]) / norm # Rotate 90 deg and normalize

        # Calculate displacement along perpendicular vector
        # Scale frequency relative to line length
        scaled_freq = self.frequency * (self.length / 100.0) # Freq per 100 pixels
        displacement = self.amplitude * np.sin(t * scaled_freq * 2 * np.pi + self.phase)

        # Add displacement to base points
        wavy_points = base_points + displacement[:, np.newaxis] * perp_vec

        return wavy_points

    def _get_draw_points(self) -> np.ndarray:
        """Returns points suitable for cv2.polylines or cv2.line."""
        if self.is_wavy and self.wavy_path_points is not None:
            return self.wavy_path_points.astype(np.int32)
        else:
            # Return start and end for straight line
            pt1 = (int(round(self.start_point[0])), int(round(self.start_point[1])))
            pt2 = (int(round(self.end_point[0])), int(round(self.end_point[1])))
            return np.array([pt1, pt2], dtype=np.int32)

    def _get_line_points(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get integer start/end points."""
        # Start/end points are already calculated in __init__
        pt1 = (int(round(self.start_point[0])), int(round(self.start_point[1])))
        pt2 = (int(round(self.end_point[0])), int(round(self.end_point[1])))
        return pt1, pt2

    def draw(self, image_data: np.ndarray) -> np.ndarray:
        """Draws the line (straight or wavy) on the image."""
        draw_points = self._get_draw_points()
        if len(draw_points) < 2: return image_data

        draw_intensity, line_type = self._get_draw_params(image_data.dtype)

        if self.is_wavy:
            cv2.polylines(
                img=image_data,
                pts=[draw_points], # Note the list wrapper
                isClosed=False,
                color=draw_intensity,
                thickness=self.thickness,
                lineType=line_type
            )
        else: # Straight line
            cv2.line(
                img=image_data,
                pt1=tuple(draw_points[0]), # cv2.line needs tuples
                pt2=tuple(draw_points[1]),
                color=draw_intensity,
                thickness=self.thickness,
                lineType=line_type
            )
        return image_data

    def generate_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generates a binary mask for the line (straight or wavy)."""
        mask = np.zeros(shape, dtype=np.uint8)
        draw_points = self._get_draw_points()
        if len(draw_points) < 2: return mask

        mask_intensity, _, line_type = self._get_mask_params() # Non-AA

        if self.is_wavy:
            cv2.polylines(
                img=mask,
                pts=[draw_points],
                isClosed=False,
                color=mask_intensity,
                thickness=self.thickness,
                lineType=line_type # LINE_8 for mask
            )
        else: # Straight line
            cv2.line(
                img=mask,
                pt1=tuple(draw_points[0]),
                pt2=tuple(draw_points[1]),
                color=mask_intensity,
                thickness=self.thickness,
                lineType=line_type # LINE_8 for mask
            )
        return mask

    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """Calculates bounding box (adjust for wavy amplitude)."""
        draw_points = self._get_draw_points()
        if len(draw_points) == 0: return (0,0,0,0)

        xmin = np.min(draw_points[:, 0])
        ymin = np.min(draw_points[:, 1])
        xmax = np.max(draw_points[:, 0])
        ymax = np.max(draw_points[:, 1])

        # Account for thickness and amplitude
        margin = self.thickness // 2
        if self.is_wavy:
             margin += int(np.ceil(abs(self.amplitude)))

        return (int(xmin - margin), int(ymin - margin), int(xmax + margin), int(ymax + margin))