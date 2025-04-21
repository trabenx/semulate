import numpy as np
import cv2
from scipy.interpolate import griddata
from typing import Dict, Any, Tuple, List

from ..base_artifact import BaseArtifact

class ElasticMeshDeform(BaseArtifact):
    """Applies smooth 'wobbly' elastic deformation using cv2.remap."""

    def apply(self, image_data: np.ndarray, **kwargs) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Applies elastic warp to image and associated masks using remap.

        Args:
            image_data (np.ndarray): The image (float32 or int).
            **kwargs: Must include 'masks' (List[np.ndarray]).

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: (modified_image, modified_masks)
        """
        masks = kwargs.get('masks')
        if masks is None:
            raise ValueError("ElasticMeshDeform requires 'masks' keyword argument.")

        rows, cols = image_data.shape[:2]

        grid_spacing = self.get_param('grid_spacing', 50) # Pixels between control points
        amplitude = self.get_param('amplitude', 5.0)     # Max pixel displacement
        smoothness = self.get_param('smoothness', 1.0)   # Controls interpolation ('linear', 'cubic')
        # Note: 'smoothness' here refers to griddata method, not physical smoothing

        # Create coarse grid points
        grid_x = np.arange(0, cols, grid_spacing)
        grid_y = np.arange(0, rows, grid_spacing)
        # Ensure borders are included
        if grid_x[-1] != cols - 1: grid_x = np.append(grid_x, cols - 1)
        if grid_y[-1] != rows - 1: grid_y = np.append(grid_y, rows - 1)

        xv, yv = np.meshgrid(grid_x, grid_y)
        grid_points = np.vstack([yv.ravel(), xv.ravel()]).T # Shape (N_points, 2) [y, x]

        # Generate random displacements at grid points
        displacements_x = np.random.uniform(-amplitude, amplitude, size=grid_points.shape[0])
        displacements_y = np.random.uniform(-amplitude, amplitude, size=grid_points.shape[0])

        # Create fine grid coordinates (every pixel)
        all_y, all_x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        pixel_points = np.vstack([all_y.ravel(), all_x.ravel()]).T # Shape (rows*cols, 2) [y, x]

        # Interpolate displacements onto the fine grid
        interp_method = 'cubic' if smoothness > 0.5 else 'linear' # Simple mapping
        try:
            map_dx = griddata(grid_points, displacements_x, pixel_points, method=interp_method, fill_value=0)
            map_dy = griddata(grid_points, displacements_y, pixel_points, method=interp_method, fill_value=0)
        except Exception as e:
             print(f"Warning: griddata interpolation failed ({e}), using linear fallback or zeros.")
             # Fallback to linear or just zeros if it fails consistently
             try:
                 map_dx = griddata(grid_points, displacements_x, pixel_points, method='linear', fill_value=0)
                 map_dy = griddata(grid_points, displacements_y, pixel_points, method='linear', fill_value=0)
             except:
                 map_dx = np.zeros(pixel_points.shape[0])
                 map_dy = np.zeros(pixel_points.shape[0])


        map_dx = map_dx.reshape((rows, cols)).astype(np.float32)
        map_dy = map_dy.reshape((rows, cols)).astype(np.float32)

        # Create the remap coordinates: map_x(y,x) = x + dx(y,x), map_y(y,x) = y + dy(y,x)
        map_x = all_x.astype(np.float32) + map_dx
        map_y = all_y.astype(np.float32) + map_dy

        # Apply the remapping
        interpolation_img = cv2.INTER_LINEAR if np.issubdtype(image_data.dtype, np.floating) else cv2.INTER_CUBIC
        interpolation_mask = cv2.INTER_NEAREST
        border_mode = cv2.BORDER_REFLECT_101 # Or other appropriate mode

        warped_image = cv2.remap(image_data, map_x, map_y,
                                 interpolation=interpolation_img,
                                 borderMode=border_mode)

        warped_masks = []
        for mask in masks:
            if mask is not None:
                warped_mask = cv2.remap(mask, map_x, map_y,
                                        interpolation=interpolation_mask,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
                warped_masks.append(warped_mask)
            else:
                warped_masks.append(None)

        return warped_image, warped_masks