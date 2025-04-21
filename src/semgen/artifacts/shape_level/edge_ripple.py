import numpy as np
import cv2
from typing import Dict, Any, Tuple

from ..base_artifact import BaseArtifact
# Consider adding Perlin noise utility if needed: from ...utils import perlin_noise

class EdgeRipple(BaseArtifact):
    """Applies small perturbations to the edges of a shape mask."""

    def apply(self, image_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Applies ripple to the boundary of a binary mask.

        Args:
            image_data (np.ndarray): The binary shape mask (uint8).
            **kwargs: Optional arguments (not strictly used here but part of signature).

        Returns:
            np.ndarray: The mask with rippled edges.
        """
        mask = image_data # Expecting mask as input
        if mask.dtype != np.uint8 or mask.ndim != 2:
             raise ValueError("EdgeRipple expects a 2D uint8 mask.")
        if np.max(mask) == 0: # Skip empty masks
             return mask

        amplitude = self.get_param('amplitude', 1.0) # Max displacement in pixels
        frequency = self.get_param('frequency', 0.5) # Ripples per perimeter unit approx
        noise_type = self.get_param('noise_type', 'sin') # 'sin' or 'perlin'

        # Find contours (edges) of the shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return mask # No contours found

        new_mask = np.zeros_like(mask)
        modified_contours = []

        for contour in contours:
            contour = contour.squeeze(axis=1)
            num_points = len(contour)
            if num_points < 3:
                modified_contours.append(contour.reshape(-1, 1, 2))
                continue

            # Calculate normals more robustly
            vec1 = np.roll(contour, -1, axis=0) - contour
            vec2 = contour - np.roll(contour, 1, axis=0)

            norm1 = np.linalg.norm(vec1, axis=1, keepdims=True)
            norm2 = np.linalg.norm(vec2, axis=1, keepdims=True)

            # Avoid division by zero
            zero_norm1 = (norm1 < 1e-6)
            zero_norm2 = (norm2 < 1e-6)
            norm1[zero_norm1] = 1.0 # Replace zero norm with 1 to avoid NaN
            norm2[zero_norm2] = 1.0

            vec1 = vec1 / norm1
            vec2 = vec2 / norm2
            # Handle cases where points were duplicates (zero vectors become zero after division)
            vec1[zero_norm1.squeeze()] = 0.0
            vec2[zero_norm2.squeeze()] = 0.0


            tangent = (vec1 + vec2) * 0.5 # Use 0.5 for average
            norm_tangent = np.linalg.norm(tangent, axis=1, keepdims=True)
            zero_norm_tangent = (norm_tangent < 1e-6)
            norm_tangent[zero_norm_tangent] = 1.0 # Avoid NaN

            tangent = tangent / norm_tangent
            tangent[zero_norm_tangent.squeeze()] = 0.0 # Handle zero tangent

            normals = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

            # Calculate displacement magnitude per point
            if noise_type == 'sin':
                perimeter_dist = np.linspace(0, num_points * frequency, num_points) # Simplified frequency scaling
                displacement_mag = amplitude * np.sin(perimeter_dist * 2 * np.pi)
            elif noise_type == 'perlin':
                # Placeholder: requires a Perlin noise implementation
                # displacement_mag = amplitude * perlin_noise_1d(np.arange(num_points) * some_scale)
                print("Warning: Perlin noise for EdgeRipple not implemented, using sin.")
                perimeter_dist = np.linspace(0, num_points * frequency, num_points)
                displacement_mag = amplitude * np.sin(perimeter_dist * 2 * np.pi)
            else: # Default to sin
                 perimeter_dist = np.linspace(0, num_points * frequency, num_points)
                 displacement_mag = amplitude * np.sin(perimeter_dist * 2 * np.pi)

            # Apply displacement along normals
            displacement = normals * displacement_mag[:, np.newaxis]

            # --- ADD Check for NaN/inf in displacement ---
            if not np.all(np.isfinite(displacement)):
                print("Warning: Non-finite values found in edge ripple displacement. Replacing with zeros.")
                displacement = np.nan_to_num(displacement, nan=0.0, posinf=0.0, neginf=0.0)
            # --- END Check ---

            new_contour_points = contour + displacement

            modified_contours.append(new_contour_points.astype(np.int32).reshape(-1, 1, 2)) 

        # Draw the modified contours onto the new mask
        # Note: This might create self-intersections or small gaps depending on amplitude
        cv2.drawContours(new_mask, modified_contours, -1, color=1, thickness=cv2.FILLED)

        # Optional: Morphological closing to fix small gaps?
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel)

        return new_mask