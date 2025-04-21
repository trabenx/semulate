import numpy as np
from typing import Dict, Any

from .base_noise import BaseNoise

class PoissonNoise(BaseNoise):
    """Applies signal-dependent Poisson noise."""

    def __init__(self, parameters: Dict[str, Any]):
         # Poisson noise is fundamentally linked to signal level, additive model often used
         parameters['mode'] = 'additive' # Or handle differently if needed
         super().__init__(parameters)
         # Lambda is complex for Poisson, often scaled relative to intensity.
         # A simpler approach uses Gaussian approx or direct scaling.
         self.scaling = self.get_param('scaling', 0.1) # Factor controlling noise strength relative to signal

    def apply(self, image_data: np.ndarray) -> np.ndarray:
        """
        Applies Poisson-like noise.
        Uses Gaussian approximation: noise ~ N(0, sqrt(signal * scaling))

        Args:
            image_data (np.ndarray): Float32 image data in [0, 1].

        Returns:
            np.ndarray: Image with added Poisson-like noise.
        """
        if not np.issubdtype(image_data.dtype, np.floating):
             print("Warning: PoissonNoise expects float input image. Converting.")
             image_data = image_data.astype(np.float32) / (np.iinfo(image_data.dtype).max if np.issubdtype(image_data.dtype, np.integer) else 1.0)

        # Ensure signal is non-negative for sqrt
        signal = np.maximum(image_data, 0)

        # Calculate per-pixel standard deviation based on signal
        sigma_map = np.sqrt(signal * self.scaling)

        # Generate Gaussian noise with varying sigma
        noise = np.random.normal(loc=0.0, scale=1.0, size=image_data.shape) * sigma_map
        noise = noise.astype(np.float32)

        noisy_image = self._apply_noise(image_data, noise)
        return noisy_image

        # --- Alternative: Direct np.random.poisson ---
        # This requires careful scaling as it returns integers.
        # vals = len(np.unique(image_data))
        # vals = 2 ** np.ceil(np.log2(vals)) # Scale image to ~photon counts
        # scaled_image = image_data * vals * self.scaling # Adjust scaling factor
        # noisy_int = np.random.poisson(scaled_image).astype(np.float32)
        # noisy_image = noisy_int / (vals * self.scaling) # Scale back
        # np.clip(noisy_image, 0.0, 1.0, out=noisy_image)
        # return noisy_image
        # ---