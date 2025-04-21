import numpy as np
from typing import Dict, Any

from .base_noise import BaseNoise

class GaussianNoise(BaseNoise):
    """Applies additive Gaussian white noise."""

    def __init__(self, parameters: Dict[str, Any]):
        # Gaussian is typically additive
        parameters['mode'] = 'additive'
        super().__init__(parameters)
        self.sigma = self.get_param('sigma', 0.05) # Std dev relative to range [0, 1]

    def apply(self, image_data: np.ndarray) -> np.ndarray:
        """
        Applies Gaussian noise.

        Args:
            image_data (np.ndarray): Float32 image data in [0, 1].

        Returns:
            np.ndarray: Image with added Gaussian noise.
        """
        if not np.issubdtype(image_data.dtype, np.floating):
             print("Warning: GaussianNoise expects float input image. Converting.")
             image_data = image_data.astype(np.float32) / (np.iinfo(image_data.dtype).max if np.issubdtype(image_data.dtype, np.integer) else 1.0)

        # Generate Gaussian noise with mean 0 and specified sigma
        noise = np.random.normal(loc=0.0, scale=self.sigma, size=image_data.shape)
        noise = noise.astype(np.float32)

        noisy_image = self._apply_noise(image_data, noise)
        return noisy_image