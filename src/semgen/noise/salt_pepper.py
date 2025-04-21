import numpy as np
import random
from typing import Dict, Any

from .base_noise import BaseNoise

class SaltPepperNoise(BaseNoise):
    """Applies Salt and Pepper (impulse) noise."""

    def __init__(self, parameters: Dict[str, Any]):
         # This noise replaces pixels, mode isn't really relevant in the same way
         super().__init__(parameters)
         self.probability = self.get_param('probability', 0.01) # Fraction of pixels affected
         self.salt_prob = 0.5 # Probability of noise being salt (vs pepper)
         self.salt_val = self.get_param('salt_intensity', 1.0) # Value for salt pixels
         self.pepper_val = self.get_param('pepper_intensity', 0.0) # Value for pepper pixels

    def apply(self, image_data: np.ndarray) -> np.ndarray:
        """
        Applies salt and pepper noise.

        Args:
            image_data (np.ndarray): Float32 image data in [0, 1].

        Returns:
            np.ndarray: Image with salt/pepper noise.
        """
        if not np.issubdtype(image_data.dtype, np.floating):
             print("Warning: SaltPepperNoise expects float input image. Converting.")
             image_data = image_data.astype(np.float32) / (np.iinfo(image_data.dtype).max if np.issubdtype(image_data.dtype, np.integer) else 1.0)

        noisy_image = image_data.copy()
        num_pixels = image_data.size
        num_noise_pixels = int(num_pixels * self.probability)

        for _ in range(num_noise_pixels):
            # Choose random pixel location
            y = random.randrange(image_data.shape[0])
            x = random.randrange(image_data.shape[1])

            # Decide salt or pepper
            if random.random() < self.salt_prob:
                noisy_image[y, x] = self.salt_val
            else:
                noisy_image[y, x] = self.pepper_val

        return noisy_image