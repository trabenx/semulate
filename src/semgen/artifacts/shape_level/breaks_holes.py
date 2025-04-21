import numpy as np
import cv2
import random
from typing import Dict, Any, Tuple

from ..base_artifact import BaseArtifact

class BreaksHoles(BaseArtifact):
    """Creates random holes or breaks within a shape mask."""

    def apply(self, image_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Adds holes to a binary mask.

        Args:
            image_data (np.ndarray): The binary shape mask (uint8).
            **kwargs: Optional args.

        Returns:
            np.ndarray: The mask with holes potentially added.
        """
        mask = image_data
        if mask.dtype != np.uint8 or mask.ndim != 2:
             raise ValueError("BreaksHoles expects a 2D uint8 mask.")
        if np.max(mask) == 0:
             return mask

        hole_count = self.get_param('hole_count', 1)
        min_size = self.get_param('min_size', 2) # pixels diameter/side
        max_size = self.get_param('max_size', 5) # pixels diameter/side
        # position_style: 'random', 'center', 'edge' (simplifying to 'random' for now)

        # Find potential locations for holes (pixels inside the shape)
        foreground_pixels = np.argwhere(mask > 0) # List of [y, x] coordinates
        if foreground_pixels.shape[0] < hole_count * min_size * min_size : # Not enough space
            return mask

        modified_mask = mask.copy()

        for _ in range(hole_count):
            if foreground_pixels.shape[0] == 0: break # Ran out of pixels

            # Choose a random location
            center_idx = random.randrange(foreground_pixels.shape[0])
            center_y, center_x = foreground_pixels[center_idx]

            # Choose a random size
            size = random.randint(min_size, max_size)
            radius = size // 2

            # Draw a black circle (hole)
            cv2.circle(modified_mask, (center_x, center_y), radius, color=0, thickness=-1)

            # Update available pixels (approximately, could be more precise)
            foreground_pixels = np.argwhere(modified_mask > 0)

        # TODO: Implement 'breaks' logic if needed (e.g., find skeleton, remove segment)

        return modified_mask