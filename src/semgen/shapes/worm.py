import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from typing import Tuple, Dict, Any

from .base_shape import BaseShape

class Worm(BaseShape):
    """
    Generates and draws a worm-like or blob shape using a smoothed random walk.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config (Dict[str, Any]): Must include 'start_point', 'intensity', 'thickness',
                                     'num_steps', 'step_size', 'smoothness'.
                                     Optional: 'rotation' (applied to the final path).
                                     'center' is calculated from the generated path.
        """
        self.start_point = np.array(config['start_point'], dtype=float)
        self.num_steps = int(config['num_steps'])
        self.step_size = float(config['step_size']) # Average step size
        self.smoothness = float(config.get('smoothness', 3)) # Spline smoothing factor (s)
        self.thickness = int(config['thickness'])

        # Generate the path first to determine center etc.
        self.path_points = self._generate_smooth_path()

        # Calculate center for BaseShape init
        center = np.mean(self.path_points, axis=0)
        config['center'] = tuple(center)
        super().__init__(config) # Handles intensity, rotation (applied later)

    def _generate_smooth_path(self) -> np.ndarray:
        """Generates points for the smoothed random walk path."""
        points = [self.start_point]
        current_pos = self.start_point.copy()
        current_angle = np.random.uniform(0, 2 * np.pi) # Initial random direction

        for _ in range(self.num_steps):
            # Add some randomness to angle (correlated random walk)
            angle_delta = np.random.normal(0, np.pi / 4) # Adjust variance for 'bendiness'
            current_angle += angle_delta

            # Add randomness to step size
            step = np.random.normal(self.step_size, self.step_size / 3)
            step = max(0, step) # Ensure step is non-negative

            # Calculate next position
            dx = step * np.cos(current_angle)
            dy = step * np.sin(current_angle)
            current_pos += np.array([dx, dy])
            points.append(current_pos.copy())

        points = np.array(points)

        # Ensure enough points for spline and remove duplicates
        if len(points) < 4:
             # Cannot compute spline, return raw points or handle error
             print(f"Warning: Worm path too short ({len(points)} points) for smoothing.")
             return points

        # Check for duplicates which break splprep
        diff = np.diff(points, axis=0)
        dist_sq = np.sum(diff**2, axis=1)
        # Keep points that are sufficiently far from the previous one
        to_keep = np.concatenate(([True], dist_sq > 1e-6))
        unique_points = points[to_keep]

        if len(unique_points) < 4:
             print(f"Warning: Worm path has too few unique points ({len(unique_points)}) for smoothing.")
             return unique_points

        # Smooth the path using B-spline interpolation
        try:
            # s=smoothness controls the tradeoff: smaller s = closer to points
            # k=3 for cubic spline
            tck, u = splprep([unique_points[:, 0], unique_points[:, 1]], s=self.smoothness, k=3, quiet=True)
            # Evaluate spline at more points for smoother curve
            u_fine = np.linspace(u.min(), u.max(), len(points) * 5) # Increase density
            x_fine, y_fine = splev(u_fine, tck)
            smooth_path = np.vstack((x_fine, y_fine)).T
            return smooth_path
        except Exception as e:
            print(f"Warning: Spline smoothing failed for worm path: {e}. Returning raw points.")
            return unique_points


    def _get_transformed_path(self) -> np.ndarray:
        """Applies rotation to the generated path points around the calculated center."""
        if self.rotation == 0:
            return self.path_points.astype(np.int32)

        cx, cy = self.center
        angle_rad = np.radians(self.rotation)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Translate path to origin, rotate, translate back
        centered_path = self.path_points - np.array([cx, cy])
        rotated_path = centered_path @ rotation_matrix.T
        transformed_path = rotated_path + np.array([cx, cy])

        return transformed_path.astype(np.int32)


    def draw(self, image_data: np.ndarray) -> np.ndarray:
        """Draws the worm path on the image."""
        path_vertices = self._get_transformed_path()
        if len(path_vertices) < 2: return image_data # Cannot draw

        draw_intensity, line_type = self._get_draw_params(image_data.dtype)

        cv2.polylines(
            img=image_data,
            pts=[path_vertices], # Note the list wrapper for polylines
            isClosed=False,      # Worm path is open
            color=draw_intensity,
            thickness=self.thickness,
            lineType=line_type
        )
        return image_data

    def generate_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generates a binary mask for the worm path."""
        mask = np.zeros(shape, dtype=np.uint8)
        path_vertices = self._get_transformed_path()
        if len(path_vertices) < 2: return mask

        mask_intensity, _, line_type = self._get_mask_params() # Non-AA

        cv2.polylines(
            img=mask,
            pts=[path_vertices],
            isClosed=False,
            color=mask_intensity,
            thickness=self.thickness,
            lineType=line_type # Use LINE_8 for binary mask
        )
        return mask

    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        # Bounding box of the transformed vertices
        vertices = self._get_transformed_path()
        if len(vertices) == 0: return (0,0,0,0)
        xmin = np.min(vertices[:, 0]) - self.thickness // 2
        ymin = np.min(vertices[:, 1]) - self.thickness // 2
        xmax = np.max(vertices[:, 0]) + self.thickness // 2
        ymax = np.max(vertices[:, 1]) + self.thickness // 2
        return (int(xmin), int(ymin), int(xmax), int(ymax))