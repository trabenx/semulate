import numpy as np
from typing import Dict, Any

from .base_noise import BaseNoise

class QuantizationNoise(BaseNoise):
    """Simulates noise from bit-depth reduction."""

    def __init__(self, parameters: Dict[str, Any]):
        # Quantization is a direct operation, mode isn't relevant
        super().__init__(parameters)
        self.n_bits = self.get_param('n_bits', 8) # Target bit depth
        self.dither_style = self.get_param('dither_style', 'none') # 'none', 'floyd_steinberg', etc. (Not implemented)

    def apply(self, image_data: np.ndarray) -> np.ndarray:
        """
        Applies quantization effect.

        Args:
            image_data (np.ndarray): Float32 image data in [0, 1].

        Returns:
            np.ndarray: Image with quantization applied (still float32).
        """
        if not np.issubdtype(image_data.dtype, np.floating):
             print("Warning: QuantizationNoise expects float input image. Converting.")
             image_data = image_data.astype(np.float32) / (np.iinfo(image_data.dtype).max if np.issubdtype(image_data.dtype, np.integer) else 1.0)

        if self.dither_style != 'none':
            print(f"Warning: Dithering style '{self.dither_style}' not implemented.")
            # TODO: Implement dithering algorithms if needed

        # Calculate number of levels
        levels = 2**self.n_bits

        # Quantize: Scale to [0, levels-1], round, scale back to [0, 1]
        quantized_image = np.round(image_data * (levels - 1)) / (levels - 1)

        # Ensure clipping just in case of rounding issues near 1.0
        np.clip(quantized_image, 0.0, 1.0, out=quantized_image)

        return quantized_image.astype(np.float32)