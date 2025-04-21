# src/semgen/artifacts/image_level/gradient_illumination.py
import numpy as np
import cv2
from ..base_artifact import BaseArtifact
from typing import Dict, Any, Tuple, List

class GradientIllumination(BaseArtifact):
     def apply(self, image_data: np.ndarray, **kwargs) -> np.ndarray:
        # Gradient affects intensity, not geometry usually
        output_image = image_data.astype(np.float32) # Work in float
        rows, cols = image_data.shape[:2]

        strength = self.get_param('strength', 0.3) # e.g., range [0,1] for max change
        style = self.get_param('style', 'linear')
        # Add params for orientation, center etc.

        gradient_map = np.zeros_like(output_image, dtype=np.float32)

        if style == 'linear':
            # TODO: Use angle/orientation parameter
            y_coords = np.linspace(-strength/2, strength/2, rows)[:, np.newaxis]
            gradient_map = np.tile(y_coords, (1, cols))
        elif style == 'radial':
             # TODO: Implement radial gradient based on center parameter
             gradient_map.fill(0) # Placeholder
        else:
             gradient_map.fill(0)

        # Apply: Additive or Multiplicative? Let's assume additive change
        output_image += gradient_map
        np.clip(output_image, 0.0, 1.0 if np.issubdtype(image_data.dtype, np.floating) else np.iinfo(image_data.dtype).max, out=output_image)

        return output_image.astype(image_data.dtype) # Pass masks through