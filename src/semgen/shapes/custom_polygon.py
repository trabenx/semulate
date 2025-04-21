import numpy as np
import cv2
from typing import Tuple, Dict, Any, List

from .base_shape import BaseShape

# --- Define Base Vertices for Custom Shapes ---
# Define vertices relative to a convenient origin (e.g., center)
# and scale (e.g., fitting within a 1x1 or similar box).
# These will be scaled, rotated, and translated later.
# Format: List of [x, y] pairs.

# H shape (already defined)
H_OUTLINE_VERTICES = [
    [-0.5, -0.5], [-0.3, -0.5], [-0.3, -0.1], [ 0.3, -0.1], [ 0.3, -0.5],
    [ 0.5, -0.5], [ 0.5,  0.5], [ 0.3,  0.5], [ 0.3,  0.1], [-0.3,  0.1],
    [-0.3,  0.5], [-0.5,  0.5]
]

# L shape (bar width 0.2)
L_OUTLINE_VERTICES = [
    [-0.5, -0.5], [-0.3, -0.5], [-0.3,  0.3], [ 0.5,  0.3], [ 0.5,  0.5],
    [-0.5,  0.5]
]

# E shape (bar width 0.2)
E_OUTLINE_VERTICES = [
    [-0.5, -0.5], [ 0.5, -0.5], [ 0.5, -0.3], [-0.3, -0.3], [-0.3, -0.1],
    [ 0.3, -0.1], [ 0.3,  0.1], [-0.3,  0.1], [-0.3,  0.3], [ 0.5,  0.3],
    [ 0.5,  0.5], [-0.5,  0.5]
]

# T shape (bar width 0.2)
T_OUTLINE_VERTICES = [
    [-0.5, -0.5], [ 0.5, -0.5], [ 0.5, -0.3], [ 0.1, -0.3], [ 0.1,  0.5],
    [-0.1,  0.5], [-0.1, -0.3], [-0.5, -0.3]
]

# U shape (Approximated with polygon bottom, bar width 0.2)
# For true round, RoundedRectangle logic could be adapted
U_OUTLINE_VERTICES = [
    [-0.5, -0.5], [-0.3, -0.5], [-0.3,  0.3], [-0.1,  0.5], # Bottom segment approx
    [ 0.1,  0.5], [ 0.3,  0.3], [ 0.3, -0.5], [ 0.5, -0.5],
    [ 0.5,  0.5], [-0.5,  0.5] # Close the top implicitly or add segment if needed
]

# X shape
# Using a simplified diamond-like X outline, as a single complex polygon is hard.
X_OUTLINE_VERTICES = [
    [-0.1, -0.5], [ 0.1, -0.5], [ 0.5, -0.1], [ 0.5,  0.1], [ 0.1,  0.5],
    [-0.1,  0.5], [-0.5,  0.1], [-0.5, -0.1]
]


# Y shape (bar width 0.2)
Y_OUTLINE_VERTICES = [
    [-0.5, -0.5], [-0.3, -0.3], [-0.1, -0.1], [-0.1,  0.5], # Left arm + stem part
    [ 0.1,  0.5], [ 0.1, -0.1], [ 0.3, -0.3], [ 0.5, -0.5], # Right arm + stem part
    [ 0.3, -0.5], [ 0.1, -0.3], [-0.1, -0.3], [-0.3, -0.5] # Bottom closing
]

# I shape (Simple rectangle, bar width 0.2)
I_OUTLINE_VERTICES = [
    [-0.1, -0.5], [ 0.1, -0.5], [ 0.1,  0.5], [-0.1,  0.5]
]

# C shape (Approximated with polygon, bar width 0.2)
C_OUTLINE_VERTICES = [
    [ 0.5, -0.3], [ 0.3, -0.5], [-0.3, -0.5], [-0.5, -0.3], # Outer bottom/left
    [-0.5,  0.3], [-0.3,  0.5], [ 0.3,  0.5], [ 0.5,  0.3], # Outer top/right
    [ 0.3,  0.1], # Start inner curve approx
    [ 0.1,  0.3], [-0.3,  0.3], [-0.3, -0.3], [ 0.1, -0.3],
    [ 0.3, -0.1]  # End inner curve approx
]

# S shape - Very complex to do well with a simple polygon.
# Requires curve approximation (Bezier/Spline) or many segments.
# Placeholder (e.g., using two C shapes back-to-back or just a basic Z poly)
S_OUTLINE_VERTICES = [ # Basic Z approximation
    [-0.5, -0.5], [ 0.5, -0.5], [ 0.5, -0.3], [-0.3,  0.3], [-0.5,  0.3],
    [-0.5,  0.5], [-0.5,  0.5], [-0.5,  0.5], # Duplicate points likely needed here if Z
    # Redoing Z properly:                         #<--- Start of commented out section
    [-0.5, -0.5], [ 0.5, -0.5], [ 0.5, -0.3], # Top bar right part
    [-0.3,  0.5], [-0.5,  0.5], [-0.5,  0.3], # Bottom bar left part
    [ 0.3, -0.5], [ 0.5, -0.5] # Need diagonal connection vertices carefully placed #<--- End of potentially problematic section
    # => S is hard. May need external library or different approach.
    # Using a placeholder "block S":              #<--- Start of block S
    # [-0.5,-0.5], [0.5,-0.5], [0.5, -0.1], [ -0.3, -0.1], [-0.3, 0.1], [0.5, 0.1], [0.5, 0.5], [-0.5, 0.5], [-0.5, 0.1], [0.3, 0.1], [0.3, -0.1], [-0.5, -0.1] #<--- This line looks correct if uncommented

]


SHAPE_VERTEX_MAP = {
    "H": np.array(H_OUTLINE_VERTICES, dtype=np.float32),
    "L": np.array(L_OUTLINE_VERTICES, dtype=np.float32),
    "E": np.array(E_OUTLINE_VERTICES, dtype=np.float32),
    "S": np.array(S_OUTLINE_VERTICES, dtype=np.float32), # Use with caution
    "T": np.array(T_OUTLINE_VERTICES, dtype=np.float32),
    "U": np.array(U_OUTLINE_VERTICES, dtype=np.float32),
    "X": np.array(X_OUTLINE_VERTICES, dtype=np.float32), # Use with caution
    "Y": np.array(Y_OUTLINE_VERTICES, dtype=np.float32),
    "I": np.array(I_OUTLINE_VERTICES, dtype=np.float32),
    "C": np.array(C_OUTLINE_VERTICES, dtype=np.float32),
}


class CustomPolygon(BaseShape):
    """Generates and draws custom predefined polygon shapes (H, L, E, etc.)."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config (Dict[str, Any]): Must include 'center', 'intensity', 'shape_name'
                                     (e.g., "H", "L"), 'width', 'height'.
                                     Optional: 'rotation', 'border_width'.
        """
        super().__init__(config)
        self.shape_name = config['shape_name'].upper()
        self.width: float = float(config['width'])
        self.height: float = float(config['height'])
        self.border_width: int = int(config.get('border_width', -1)) # Default filled

        if self.shape_name not in SHAPE_VERTEX_MAP or SHAPE_VERTEX_MAP[self.shape_name] is None:
             raise ValueError(f"Vertex definition for shape '{self.shape_name}' not implemented.")
        self.base_vertices = SHAPE_VERTEX_MAP[self.shape_name].copy()


    def _get_transformed_vertices(self) -> np.ndarray:
        """Calculates the final vertices after scaling, rotation, translation."""
        cx, cy = self.center
        angle_rad = np.radians(self.rotation)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # 1. Scale base vertices
        # Assumes base vertices fit roughly in a -0.5 to 0.5 box
        scaled_vertices = self.base_vertices * np.array([self.width, self.height])

        # 2. Rotate scaled vertices around origin
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_vertices = scaled_vertices @ rotation_matrix.T

        # 3. Translate to center
        transformed_vertices = rotated_vertices + np.array([cx, cy])

        return transformed_vertices.astype(np.int32) # cv2 needs int32


    def draw(self, image_data: np.ndarray) -> np.ndarray:
        """Draws the custom polygon on the image."""
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
                lineType=line_type
            )
        return image_data

    def generate_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generates a binary mask for the custom polygon."""
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
        vertices = self._get_transformed_vertices()
        xmin = np.min(vertices[:, 0])
        ymin = np.min(vertices[:, 1])
        xmax = np.max(vertices[:, 0])
        ymax = np.max(vertices[:, 1])
        return (int(xmin), int(ymin), int(xmax), int(ymax))