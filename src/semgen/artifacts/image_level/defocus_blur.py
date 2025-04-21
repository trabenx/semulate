# src/semgen/artifacts/image_level/defocus_blur.py
import numpy as np
import cv2
from ..base_artifact import BaseArtifact
from typing import Dict, Any, Tuple, List

class DefocusBlur(BaseArtifact):
    def apply(self, image_data: np.ndarray, **kwargs) -> np.ndarray:
        masks = kwargs.get('masks')
        if masks is None: # Defocus doesn't usually warp geometry
             print("Warning: DefocusBlur received no masks, only blurring image.")
             masks_out = []
        else:
            masks_out = masks # Pass masks through unchanged

        rows, cols = image_data.shape[:2]
        focal_plane_norm = self.get_param('focal_plane', 0.5) # 0=top, 1=bottom
        max_radius = self.get_param('max_radius', 3.0) # Max blur sigma/radius
        gradient_axis = self.get_param('gradient_axis', 'vertical')

        if max_radius <= 0: return image_data, masks_out # No blur

        # Create spatially varying blur map (sigma map)
        sigma_map = np.zeros_like(image_data, dtype=np.float32)
        if gradient_axis == 'vertical':
            y_coords = np.arange(rows)[:, np.newaxis]
            distance = np.abs(y_coords - (focal_plane_norm * rows))
            max_dist = max(focal_plane_norm * rows, (1.0 - focal_plane_norm) * rows)
            sigma_map = (distance / max_dist) * max_radius if max_dist > 0 else 0
        # TODO: Implement horizontal, radial gradient axis for sigma_map

        # Apply blur - This is tricky! cv2.GaussianBlur doesn't take sigma map.
        # Need per-pixel filtering or approximation (e.g., blur tiers)
        # Approximation: Apply uniform blur with average sigma for now
        print("Warning: Spatially varying defocus blur not fully implemented. Applying uniform blur.")
        avg_sigma = max_radius / 2.0 # Rough average
        ksize = int(max(3, 2 * np.ceil(3 * avg_sigma) + 1))
        
        if avg_sigma <= 0: return image_data # No blur if sigma is zero or less

        print("Warning: Spatially varying defocus blur not fully implemented. Applying uniform blur.")
        blurred_image = cv2.GaussianBlur(image_data, (ksize, ksize), avg_sigma)

        # Replace with proper implementation later if needed

        return blurred_image.astype(image_data.dtype)