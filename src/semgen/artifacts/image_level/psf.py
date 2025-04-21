import numpy as np
import cv2
from typing import Dict, Any

from ..base_artifact import BaseArtifact

class ProbePSF(BaseArtifact):
    """Applies instrument Point Spread Function blur (convolution)."""

    def apply(self, image_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Applies PSF blur to the image.

        Args:
            image_data (np.ndarray): The image (float32 or int).
            **kwargs: Optional args.

        Returns:
            np.ndarray: Modified image.
        """
        sigma_x = self.get_param('sigma_x', 1.0)
        sigma_y = self.get_param('sigma_y', sigma_x) # Default to symmetric if y not given
        angle = self.get_param('astig_angle', 0.0) # Degrees
        kernel_type = self.get_param('kernel_type', 'gaussian') # 'gaussian', 'airy' (not impl), 'custom' (not impl)
        
        if sigma_x <= 0 and sigma_y <= 0: return image_data

        if kernel_type != 'gaussian':
            print(f"Warning: Kernel type '{kernel_type}' not implemented for PSF. Using Gaussian.")

        # Determine kernel size (e.g., 3x sigma)
        ksize_x = int(max(3, 2 * np.ceil(3 * sigma_x) + 1)) # Odd size >= 3
        ksize_y = int(max(3, 2 * np.ceil(3 * sigma_y) + 1)) # Odd size >= 3

        # Create base Gaussian kernel (aligned with axes)
        # Using getGaussianKernel is usually for separable kernels.
        # For rotated elliptical, build 2D kernel directly.

        # Create 2D Gaussian kernel
        kernel = cv2.getGaussianKernel(ksize_x, sigma_x) @ cv2.getGaussianKernel(ksize_y, sigma_y).T
        # This creates an axis-aligned elliptical Gaussian

        if abs(sigma_x - sigma_y) > 1e-3 or angle != 0:
            # Need to handle rotation for elliptical kernel
            # This is complex. Simpler approach: Apply axis-aligned blur using separate sigmas?
            # Or use cv2.filter2D with a manually constructed rotated kernel.

            # Approximation: Use cv2.GaussianBlur with potentially different sigmas
            # Note: cv2.GaussianBlur doesn't support angle directly.
            # We might need a custom kernel generation or accept axis-aligned elliptical blur.
            # Let's use axis-aligned for now.
            ksize = (ksize_x, ksize_y) # Use separate sizes maybe? No, GaussianBlur needs one ksize tuple. Use max.
            ksize_use = (max(ksize_x, ksize_y), max(ksize_x, ksize_y))

            # Use sigmaX and sigmaY for GaussianBlur if available in OpenCV version, otherwise use average?
            # Most versions support sigmaX, sigmaY is derived or set to 0 for symmetric
            # Let's just use sigma_x and let GaussianBlur handle it, or use filter2D if needed.
            # Using GaussianBlur:
            blurred_image = cv2.GaussianBlur(image_data, ksize=ksize_use, sigmaX=sigma_x, sigmaY=sigma_y)
            # This DOES NOT handle the astigmatism angle correctly.

            # TODO: Implement rotated elliptical kernel generation and cv2.filter2D for proper astigmatism.
            if angle != 0:
                 print("Warning: PSF Astigmatism angle not fully implemented with cv2.GaussianBlur. Applying axis-aligned blur.")

        else: # Symmetric Gaussian
            ksize = (ksize_x, ksize_x) # Make symmetric
            blurred_image = cv2.GaussianBlur(image_data, ksize=ksize, sigmaX=sigma_x)


        return blurred_image