import numpy as np
import cv2
import random
from typing import Dict, Any

from ..base_artifact import BaseArtifact

class Charging(BaseArtifact):
    """Simulates charging artifacts like halos and streaks."""

    def apply(self, image_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Adds charging halos and/or streaks to the image.

        Args:
            image_data (np.ndarray): The image (float32 or int).
            **kwargs: Optional args.

        Returns:
            np.ndarray: Modified image.
        """
        rows, cols = image_data.shape[:2]
        output_image = image_data.astype(np.float32) # Work in float

        halo_intensity = self.get_param('halo_intensity', 0.1) # Additive brightness
        halo_radius = self.get_param('halo_radius', 20) # Pixels
        streak_probability = self.get_param('streak_probability', 0.1)
        streak_length = self.get_param('streak_length', 100) # Pixels
        streak_direction = self.get_param('streak_direction', 0) # Degrees
        # num_halos/streaks could also be parameters

        # --- Add Halos ---
        # Simplified: Add one halo at a random bright-ish spot or just random location
        if halo_intensity > 0 and halo_radius > 0:
             # Find potential center (e.g., percentile brightness or random)
             # Random location for simplicity:
             center_x = random.randint(0, cols - 1)
             center_y = random.randint(0, rows - 1)

             # Create halo effect (e.g., blurred Gaussian blob)
             halo_map = np.zeros_like(output_image, dtype=np.float32)
             cv2.circle(halo_map, (center_x, center_y), int(halo_radius), color=halo_intensity, thickness=-1)

             # Blur the halo
             blur_ksize = int(max(3, 2 * np.ceil(halo_radius / 2) + 1)) # Odd size
             halo_map = cv2.GaussianBlur(halo_map, (blur_ksize, blur_ksize), halo_radius / 3)

             # Add halo to image (additive blending)
             output_image += halo_map

        # --- Add Streaks ---
        if random.random() < streak_probability and streak_length > 0:
             # Choose start point (e.g., random edge or random point)
             start_x = random.randint(0, cols - 1)
             start_y = random.randint(0, rows - 1)

             # Calculate end point based on direction and length
             angle_rad = np.radians(streak_direction)
             end_x = int(round(start_x + streak_length * np.cos(angle_rad)))
             end_y = int(round(start_y + streak_length * np.sin(angle_rad)))

             # Create streak effect (e.g., bright line, possibly blurred)
             streak_map = np.zeros_like(output_image, dtype=np.float32)
             streak_intensity = halo_intensity * 0.8 # Link intensity? Or separate param?
             streak_thickness = int(max(1, halo_radius / 5)) # Link thickness?

             cv2.line(streak_map, (start_x, start_y), (end_x, end_y),
                      color=streak_intensity, thickness=streak_thickness, lineType=cv2.LINE_AA)

             # Optional: Blur the streak slightly?
             blur_ksize_s = int(max(3, 2 * np.ceil(streak_thickness) + 1))
             if blur_ksize_s > 1:
                  streak_map = cv2.GaussianBlur(streak_map, (blur_ksize_s, blur_ksize_s), streak_thickness / 3)

             output_image += streak_map

        # Clip result to valid range and convert back to original type if needed
        np.clip(output_image, 0.0, 1.0 if np.issubdtype(image_data.dtype, np.floating) else np.iinfo(image_data.dtype).max, out=output_image)

        return output_image.astype(image_data.dtype)