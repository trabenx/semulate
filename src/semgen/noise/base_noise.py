import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseNoise(ABC):
    """
    Abstract base class for all noise types applied to the SEM image.
    """
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize the noise generator with its specific parameters.

        Args:
            parameters (Dict[str, Any]): Dictionary containing parameters specific
                                         to this noise instance (e.g., sigma, lambda,
                                         probability, amplitude), already sampled from
                                         the configured ranges by the raffle mechanism.
                                         Should also include 'mode' ('additive' or 'multiplicative').
        """
        self.params = parameters
        self.mode = self.get_param('mode', 'additive').lower() # Default to additive

    @abstractmethod
    def apply(self, image_data: np.ndarray) -> np.ndarray:
        """
        Apply the noise effect to the image data.

        Args:
            image_data (np.ndarray): The image data, ideally float32 in [0, 1] range.

        Returns:
            np.ndarray: The image data with noise applied.
        """
        pass

    def get_param(self, key: str, default: Any = None) -> Any:
        """Helper to safely get a parameter value."""
        return self.params.get(key, default)

    def _apply_noise(self, image_data: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """Applies the generated noise based on the mode."""
        if self.mode == 'additive':
            noisy_image = image_data + noise
        elif self.mode == 'multiplicative':
            # Multiplicative noise often applied as: image * (1 + noise) or image * noise_map
            # Assuming noise represents multiplicative factor around 1
            noisy_image = image_data * (1.0 + noise)
            # Or if noise is directly the multiplier map:
            # noisy_image = image_data * noise
        else:
             print(f"Warning: Unknown noise mode '{self.mode}'. Applying additively.")
             noisy_image = image_data + noise

        # Clip to valid range [0, 1] for float images
        # Note: Clipping might alter noise statistics, especially for strong noise.
        np.clip(noisy_image, 0.0, 1.0, out=noisy_image)
        return noisy_image