import numpy as np
import cv2
from typing import Dict, Any

from ..base_artifact import BaseArtifact
# Assumes a Perlin noise utility exists:
# from ...utils.noise_utils import generate_perlin_noise_2d

class TopographicShading(BaseArtifact):
    """Applies brightness modulation based on a synthetic height map slope."""

    def apply(self, image_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Applies topographic shading effect.

        Args:
            image_data (np.ndarray): The image (float32 or int).
            **kwargs: Optional args like layer info if source is 'derived'.

        Returns:
            np.ndarray: Modified image.
        """
        rows, cols = image_data.shape[:2]
        output_image = image_data.astype(np.float32) # Work in float

        contrast = self.get_param('height_contrast', 0.3) # Modulation strength
        slope_angle_deg = self.get_param('slope_angle', 45.0) # Light source direction
        source = self.get_param('height_map_source', 'random_perlin')

        # 1. Generate or get the height map
        if source == 'derived':
            # Requires access to layer structure - complex integration needed
            print("Warning: 'derived' height map source not implemented for TopographicShading.")
            # Fallback: generate a simple gradient or flat map?
            height_map = np.zeros_like(output_image, dtype=np.float32) # Flat map = no effect
        elif source == 'random_perlin':
            try:
                # Placeholder call - requires actual implementation
                # from ...utils.noise_utils import generate_perlin_noise_2d
                # Assume generate_perlin_noise_2d exists and returns noise in [-1, 1] or [0, 1]
                # octaves = 4
                # persistence = 0.5
                # lacunarity = 2.0
                # scale = 50.0 # Controls feature size
                # height_map = generate_perlin_noise_2d((rows, cols), (scale, scale), octaves, persistence, lacunarity)
                print("Warning: Perlin noise generation not implemented. Using simple gradient as height map.")
                # Simple gradient fallback
                y_coords, x_coords = np.mgrid[0:rows, 0:cols]
                height_map = (y_coords.astype(np.float32) / rows) * 2.0 - 1.0 # Range [-1, 1]

            except ImportError:
                 print("Warning: Perlin noise utility not found. Using simple gradient as height map.")
                 y_coords, x_coords = np.mgrid[0:rows, 0:cols]
                 height_map = (y_coords.astype(np.float32) / rows) * 2.0 - 1.0
        else:
             print(f"Warning: Unknown height_map_source '{source}'. No shading applied.")
             height_map = np.zeros_like(output_image, dtype=np.float32)

        # 2. Calculate gradient (slope) of the height map
        grad_y = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
        grad_x = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)

        # 3. Calculate shading factor based on dot product with light direction
        light_angle_rad = np.radians(slope_angle_deg)
        light_vec = np.array([np.cos(light_angle_rad), np.sin(light_angle_rad)])

        # Normalize gradient vectors (where magnitude > 0)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        # Avoid division by zero
        inv_magnitude = np.divide(1.0, magnitude, where=magnitude != 0)

        norm_grad_x = grad_x * inv_magnitude
        norm_grad_y = grad_y * inv_magnitude

        # Dot product: shading = light_vec[0]*norm_grad_x + light_vec[1]*norm_grad_y
        # Ranges from -1 (opposite light) to +1 (towards light)
        shading_factor = light_vec[0] * norm_grad_x + light_vec[1] * norm_grad_y

        # 4. Apply shading modulation
        # Map shading_factor [-1, 1] to a multiplier [1-contrast, 1+contrast] (approx)
        # Or simply add scaled shading_factor: Intensity = Intensity + Shading * Contrast
        modulation = shading_factor * contrast
        output_image += modulation # Additive shading

        # Clip result
        np.clip(output_image, 0.0, 1.0 if np.issubdtype(image_data.dtype, np.floating) else np.iinfo(image_data.dtype).max, out=output_image)

        return output_image.astype(image_data.dtype)