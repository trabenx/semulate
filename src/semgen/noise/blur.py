import numpy as np
import cv2
from typing import Dict, Any

from .base_noise import BaseNoise

class BlurNoise(BaseNoise):
    """Applies simple spatial blur (Gaussian or potentially motion)."""

    def __init__(self, parameters: Dict[str, Any]):
        # Blur is a convolution, mode isn't relevant
        super().__init__(parameters)
        self.kernel_size = self.get_param('kernel_size', 3) # Size of blur kernel (must be odd)
        self.sigma = self.get_param('sigma', 0) # Gaussian sigma, if 0, calculated from kernel_size
        # self.direction = self.get_param('direction', 0) # Angle for motion blur (Not implemented)
        self.blur_type = self.get_param('blur_type', 'gaussian') # 'gaussian', 'motion' (Not implemented)

    def apply(self, image_data: np.ndarray) -> np.ndarray:
        """
        Applies spatial blur.

        Args:
            image_data (np.ndarray): Float32 or Integer image data.

        Returns:
            np.ndarray: Blurred image.
        """
        if self.blur_type == 'motion':
            print("Warning: Motion blur not implemented. Using Gaussian blur.")
            # TODO: Implement motion blur kernel generation

        ksize = int(self.kernel_size)
        # Ensure kernel size is odd
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(1, ksize) # Must be at least 1

        # Apply Gaussian Blur
        # Note: sigmaY = sigmaX if sigmaY is 0
        blurred_image = cv2.GaussianBlur(image_data, ksize=(ksize, ksize), sigmaX=self.sigma)

        return blurred_image