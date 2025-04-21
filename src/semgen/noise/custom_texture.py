import numpy as np
from typing import Dict, Any

from .base_noise import BaseNoise
from .sem_texture import SEMTextureNoise # Reuse logic

class CustomTexturalNoise(SEMTextureNoise):
    """
    Applies low-frequency textural noise (e.g., Perlin/Simplex).
    Inherits from SEMTextureNoise as the mechanism is similar.
    """
    def __init__(self, parameters: Dict[str, Any]):
        # Ensure parameters expected by SEMTextureNoise are present or defaulted
        parameters.setdefault('contrast', parameters.get('amplitude', 0.15))
        parameters.setdefault('frequency', parameters.get('frequency', 2.0)) # Lower frequency for custom texture
        parameters.setdefault('style', parameters.get('style', 'perlin'))
        parameters.setdefault('mode', parameters.get('mode', 'multiplicative'))
        super().__init__(parameters)

    # apply method is inherited from SEMTextureNoise