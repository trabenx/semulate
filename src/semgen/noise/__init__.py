from typing import Dict, Any
from .base_noise import BaseNoise
from .gaussian import GaussianNoise
from .poisson import PoissonNoise
from .salt_pepper import SaltPepperNoise
from .sem_texture import SEMTextureNoise
from .quantization import QuantizationNoise
from .blur import BlurNoise
from .custom_texture import CustomTexturalNoise
# Import FixedPatternNoise when created

# Optional: Noise Factory
NOISE_CLASS_MAP = {
    "gaussian": GaussianNoise,
    "poisson": PoissonNoise,
    "salt_pepper": SaltPepperNoise,
    "sem_texture": SEMTextureNoise,
    "quantization": QuantizationNoise,
    "blur": BlurNoise,
    "custom_texture": CustomTexturalNoise,
    # "fixed_pattern": FixedPatternNoise,
}

def create_noise(noise_config: Dict[str, Any]) -> BaseNoise:
    """Factory function to create noise instances."""
    noise_type = noise_config.get('noise_type', noise_config.get('type', '')).lower()
    if not noise_type:
        raise ValueError("Noise configuration must include 'noise_type' or 'type'.")

    NoiseClass = NOISE_CLASS_MAP.get(noise_type)
    if NoiseClass is None:
        raise ValueError(f"Unknown noise type: '{noise_type}'")

    # Parameters specific to the noise type are passed directly
    try:
        return NoiseClass(parameters=noise_config)
    except Exception as e:
        raise RuntimeError(f"Error creating noise '{noise_type}': {e}") from e


__all__ = [
    "BaseNoise",
    "GaussianNoise",
    "PoissonNoise",
    "SaltPepperNoise",
    "SEMTextureNoise",
    "QuantizationNoise",
    "BlurNoise",
    "CustomTexturalNoise",
    # "FixedPatternNoise",
    "create_noise", # Expose factory
]