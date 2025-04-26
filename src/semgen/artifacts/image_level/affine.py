import numpy as np
import cv2
from typing import Dict, Any, Tuple, List

from ..base_artifact import BaseArtifact

class AffineTransform(BaseArtifact):
    """Applies affine transformations (scale, rotate, shear, translate)."""

    def apply(self, image_data: np.ndarray, **kwargs) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Applies affine warp to image and associated masks.

        Args:
            image_data (np.ndarray): The image (float32 or int).
            **kwargs: Must include 'masks' (List[np.ndarray]).

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: (modified_image, modified_masks)
        """
        masks = kwargs.get('masks')
        if masks is None:
            raise ValueError("AffineTransform requires 'masks' keyword argument.")

        rows, cols = image_data.shape[:2]
        center = (cols / 2, rows / 2)

        angle = self.get_param('angle', 0.0)
        scale = self.get_param('scale', 1.0)
        # Shear needs separate handling or a full 2x3 matrix definition
        shear_x = self.get_param('shear_x', 0.0) # Simple shear factor along x
        tx = self.get_param('translate_x', 0.0) # Pixels
        ty = self.get_param('translate_y', 0.0) # Pixels

        # 1. Rotation and Scaling matrix
        M_rot_scale = cv2.getRotationMatrix2D(center, angle, scale)

        # 2. Add Shear - modify the rotation/scale matrix
        # Shearing matrix (horizontal): [[1, shear_x, 0], [0, 1, 0]]
        M_shear = np.float32([[1, shear_x, 0], [0, 1, 0]])
        # Combine shear with rotation/scale. Need 3x3 matrices for multiplication.
        M_rot_scale_3x3 = np.vstack([M_rot_scale, [0, 0, 1]])
        M_shear_3x3 = np.vstack([M_shear, [0, 0, 1]])
        M_combined_3x3 = M_shear_3x3 @ M_rot_scale_3x3 # Apply rot/scale first, then shear
        M_combined = M_combined_3x3[0:2, :] # Back to 2x3

        # 3. Add Translation
        M_combined[0, 2] += tx
        M_combined[1, 2] += ty

        # Apply the transformation
        # Use INTER_LINEAR for image, INTER_NEAREST for masks
        if np.issubdtype(image_data.dtype, np.integer):
             interp_img = cv2.INTER_NEAREST # Use NEAREST for integer/label images
             border_mode_img = cv2.BORDER_CONSTANT
             border_val_img = 0
        else: # Floating point image data
             interp_img = cv2.INTER_LINEAR # Or INTER_CUBIC
             border_mode_img = cv2.BORDER_REFLECT_101 # Reflect often better for grayscale
             border_val_img = 0 # Not used for reflect

        # Always use NEAREST for masks in the list
        interp_mask = cv2.INTER_NEAREST
        border_mode_mask = cv2.BORDER_CONSTANT
        border_val_mask = 0

        warped_image = cv2.warpAffine(image_data, M_combined, (cols, rows),
                                      flags=interp_img,
                                      borderMode=border_mode_img,
                                      borderValue=border_val_img)

        warped_masks = []
        for mask in masks:
            if mask is not None:
                 warped_mask = cv2.warpAffine(mask, M_combined, (cols, rows),
                                             flags=interp_mask,
                                             borderMode=border_mode_mask,
                                             borderValue=border_val_mask)
                 warped_masks.append(warped_mask)
            else:
                 warped_masks.append(None)

        return warped_image, warped_masks