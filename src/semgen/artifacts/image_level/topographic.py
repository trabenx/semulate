import numpy as np
import cv2

from ..base_artifact import BaseArtifact
from ...utils.noise_utils import generate_procedural_noise_2d
from typing import Dict, Any, Tuple, List

class TopographicShading(BaseArtifact):
    """Applies brightness modulation based on a synthetic height map slope."""

    def apply(self, image_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Applies topographic shading effect.

        Args:
            image_data (np.ndarray): The image (float32 or int).
            **kwargs: Can include 'cumulative_masks' (List[np.ndarray]) if source is 'derived',
                      and 'rng' (random.Random) if needed for Perlin seed.

        Returns:
            np.ndarray: Modified image.
        """
        rows, cols = image_data.shape[:2]
        output_image = image_data.astype(np.float32) # Work in float

        contrast = self.get_param('height_contrast', 0.3)
        slope_angle_deg = self.get_param('slope_angle', 45.0)
        source = self.get_param('height_map_source', 'random_perlin')
        height_map = np.zeros_like(output_image, dtype=np.float32)

        # 1. Generate or get the height map
        if source == 'derived':
            cumulative_masks = kwargs.get('cumulative_masks')
            if cumulative_masks and isinstance(cumulative_masks, list):
                print(f"DEBUG Topo: Deriving height map from {len(cumulative_masks)} cumulative masks.")
                # Simple approach: Each subsequent cumulative mask adds a fixed height offset.
                # Normalize height to approx [0, 1] range.
                height_step = 1.0 / max(1, len(cumulative_masks))
                current_height = 0.0
                for mask in cumulative_masks:
                     if mask is not None:
                          height_map[mask > 0] = current_height + height_step # Pixels in this layer get higher value
                     current_height += height_step
                # Optionally add a small amount of blur to smooth step edges
                height_map = cv2.GaussianBlur(height_map, (5, 5), 0.5)
            else:
                 print("Warning: 'derived' height map source selected but no 'cumulative_masks' provided. Using flat map.")
                 height_map.fill(0.0) # Flat map = no effect

        elif source == 'random_perlin':
            # Get Perlin parameters (can be randomized via config)
            scale_param = self.get_param('perlin_frequency', 5.0) # Features per image width
            scale = cols / max(1.0, scale_param)
            octaves = int(self.get_param('perlin_octaves', 4))
            persistence = float(self.get_param('perlin_persistence', 0.5))
            lacunarity = float(self.get_param('perlin_lacunarity', 2.0))
            rng = kwargs.get('rng') # Need RNG for seed
            base_seed = rng.randint(0, 100000) if rng else 0

            try:
                # Generate noise in [0, 1] range to represent height
                height_map = generate_procedural_noise_2d(
                    shape=(rows, cols),
                    scale=scale, octaves=octaves, persistence=persistence,
                    lacunarity=lacunarity, base_seed=base_seed,
                    noise_type='perlin', normalize_range=(0.0, 1.0)
                )
            except Exception as e:
                 print(f"Warning: Failed to generate Perlin noise for height map ({e}). Using flat map.")
                 height_map.fill(0.0)
        else:
            print(f"Warning: Unknown height_map_source '{source}'. No shading applied.")
            height_map.fill(0.0)
             
        # --- Steps 2, 3, 4: Calculate gradient and apply shading (remain the same) ---
        if np.max(height_map) - np.min(height_map) < 1e-6: # Skip if flat map
            print("DEBUG Topo: Height map is flat, skipping shading.")
            return image_data # Return original image

        # 2. Calculate gradient (slope) of the height map
        grad_y = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
        grad_x = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)

        # 3. Calculate shading factor based on dot product with light direction
        light_angle_rad = np.radians(slope_angle_deg)
        light_vec = np.array([np.cos(light_angle_rad), np.sin(light_angle_rad)])

        # Normalize gradient vectors (where magnitude > 0)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        inv_magnitude = np.divide(1.0, magnitude, where=magnitude > 1e-6, out=np.zeros_like(magnitude)) # Handle zero magnitude

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