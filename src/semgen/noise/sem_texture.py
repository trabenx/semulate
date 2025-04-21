import numpy as np
import random # Need for base_seed
from typing import Dict, Any

from .base_noise import BaseNoise
from ..utils.noise_utils import generate_procedural_noise_2d # Import utility

# Assumes a Perlin noise utility exists:
# from ..utils.noise_utils import generate_perlin_noise_2d

class SEMTextureNoise(BaseNoise):
    """Applies fine granular structured noise, simulating SEM texture."""

    def __init__(self, parameters: Dict[str, Any]):
         parameters['mode'] = parameters.get('mode', 'multiplicative').lower()
         super().__init__(parameters)
         self.contrast = self.get_param('contrast', 0.1)
         # Config 'frequency' means features per image width (approx)
         self.frequency = self.get_param('frequency', 15.0)
         self.style = self.get_param('style', 'perlin')
         # Add other noise params if needed from config
         self.octaves = int(self.get_param('octaves', 3))
         self.persistence = float(self.get_param('persistence', 0.4))
         self.lacunarity = float(self.get_param('lacunarity', 2.2))


    def apply(self, image_data: np.ndarray) -> np.ndarray:
        """
        Applies SEM-like texture noise using Perlin/Simplex.

        Args:
            image_data (np.ndarray): Float32 image data in [0, 1].

        Returns:
            np.ndarray: Image with texture noise.
        """
        if not np.issubdtype(image_data.dtype, np.floating):
             print("Warning: SEMTextureNoise expects float input image. Converting.")
             image_data = image_data.astype(np.float32) / (np.iinfo(image_data.dtype).max if np.issubdtype(image_data.dtype, np.integer) else 1.0)

        rows, cols = image_data.shape[:2]
        # Convert frequency (features/image) to scale (pixels/feature)
        scale = cols / max(1.0, self.frequency)
        base_seed = random.randint(0, 100000) # Different seed for each application

        if self.style in ['perlin', 'simplex']:
            # Generate noise centered around 0, range approx [-1, 1] for modulation
            noise_map = generate_procedural_noise_2d(
                shape=(rows, cols),
                scale=scale,
                octaves=self.octaves,
                persistence=self.persistence,
                lacunarity=self.lacunarity,
                base_seed=base_seed,
                noise_type=self.style,
                normalize_range=(-1.0, 1.0) # Get noise centered at 0
            )
            # Scale by contrast
            noise_map *= self.contrast
        else:
             print(f"Warning: Noise style '{self.style}' not implemented for SEM Texture. Using scaled random noise.")
             noise_map = (np.random.rand(rows, cols) * 2.0 - 1.0) * self.contrast

        noise_map = noise_map.astype(np.float32)

        # Apply based on mode (using the parent class helper)
        noisy_image = self._apply_noise(image_data, noise_map)

        return noisy_image