# src/semgen/artifacts/image_level/defocus_blur.py
import numpy as np
import cv2
from ..base_artifact import BaseArtifact
from typing import Dict, Any, Tuple, List

class DefocusBlur(BaseArtifact):
    def apply(self, image_data: np.ndarray, **kwargs) -> np.ndarray:
        rows, cols = image_data.shape[:2]
        focal_plane_norm = self.get_param('focal_plane', 0.5)
        max_radius = self.get_param('max_radius', 3.0) # Max blur sigma
        gradient_axis = self.get_param('gradient_axis', 'vertical')
        num_levels = int(self.get_param('num_blur_levels', 5)) # Make levels configurable
        num_levels = max(2, num_levels) # Need at least 2 levels for interpolation

        if max_radius <= 0.1: return image_data # Skip if max blur negligible

        # --- Create spatially varying sigma map (as before) ---
        sigma_map = np.zeros_like(image_data, dtype=np.float32)
        y_coords, x_coords = np.mgrid[0:rows, 0:cols]

        # --- Calculating distance and max_dist based on gradient_axis ---
        if gradient_axis == 'vertical':
            focal_line_y = focal_plane_norm * (rows - 1)
            distance = np.abs(y_coords - focal_line_y)
            max_dist = max(focal_line_y, (rows - 1) - focal_line_y)
        elif gradient_axis == 'horizontal':
            focal_line_x = focal_plane_norm * (cols - 1)
            distance = np.abs(x_coords - focal_line_x)
            max_dist = max(focal_line_x, (cols - 1) - focal_line_x)
        elif gradient_axis == 'radial':
            center_x, center_y = cols / 2.0, rows / 2.0
            dist_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
            distance = np.sqrt(dist_sq)
            max_dist_img = np.sqrt((cols/2.0)**2 + (rows/2.0)**2)
            if focal_plane_norm < 0.5:
                 max_dist = max_dist_img
                 sigma_map = (distance / max_dist) * max_radius * (1.0 - focal_plane_norm * 2.0) if max_dist > 0 else 0
                 distance = None
            else:
                 max_dist = max_dist_img * (1.0 - focal_plane_norm)
                 sigma_map = np.maximum(0, (1.0 - distance / max_dist)) * max_radius if max_dist > 0 else 0
                 distance = None
        else:
             print(f"Warning: Unknown gradient_axis '{gradient_axis}' for DefocusBlur.")
             max_dist = 0
             distance = None

        if distance is not None:
            if max_dist > 0:
                sigma_map = (distance / max_dist) * max_radius
            else:
                 sigma_map.fill(0)
        sigma_map = np.maximum(0, sigma_map) # Ensure non-negative sigma
        # --- End sigma map calculation ---


        # --- Apply Spatially Varying Blur with Interpolation ---
        blurred_image = np.zeros_like(image_data, dtype=np.float32)
        min_sigma_in_map = np.min(sigma_map)
        max_sigma_in_map = np.max(sigma_map)

        # Define the sigma values for our discrete blur levels
        # Include 0 sigma (original image) implicitly or explicitly? Let's include it if min_sigma > 0
        sigma_levels = np.linspace(min_sigma_in_map, max_sigma_in_map, num_levels)
        # Make sure 0 is included if it's relevant (i.e., if some part is in focus)
        if min_sigma_in_map > 0.1: # Add level near 0 if min is not already close
             sigma_levels = np.insert(sigma_levels, 0, 0.0)
             num_levels = len(sigma_levels) # Update level count

        # Pre-calculate blurred versions
        blurred_layers = []
        for sigma in sigma_levels:
            if sigma < 0.1: # Treat small sigma as no blur
                 blurred_layers.append(image_data.astype(np.float32))
                 continue
            ksize = int(max(3, 2 * np.ceil(3 * sigma) + 1))
            blurred_layers.append(cv2.GaussianBlur(image_data, (ksize, ksize), sigma).astype(np.float32))

        # --- Linear Interpolation Blending ---
        # Calculate where each pixel's sigma falls relative to the discrete levels
        # Find the indices of the two nearest levels for each pixel sigma
        # 'indices' will be the index of the level *below* the pixel's sigma
        indices = np.searchsorted(sigma_levels, sigma_map, side='right') - 1
        indices = np.clip(indices, 0, num_levels - 2) # Ensure index is valid for level_0

        # Get the sigma values for the levels below (level_0) and above (level_1) each pixel
        sigma_0 = sigma_levels[indices]
        sigma_1 = sigma_levels[indices + 1]

        # Calculate the interpolation weight (alpha)
        # Avoid division by zero where sigma_0 == sigma_1
        delta_sigma = sigma_1 - sigma_0
        # Weight = 0 if pixel_sigma <= sigma_0, 1 if pixel_sigma >= sigma_1
        # Weight = (pixel_sigma - sigma_0) / (sigma_1 - sigma_0) otherwise
        alpha = np.divide(sigma_map - sigma_0, delta_sigma, where=delta_sigma > 1e-6)
        alpha = np.clip(alpha, 0.0, 1.0) # Ensure weight is between 0 and 1

        # Get the corresponding blurred image layers
        # We need efficient indexing based on the `indices` array
        # This is tricky with direct numpy indexing if blurred_layers is a list
        # Alternative: Loop through levels or use more advanced indexing if possible

        # Simpler (potentially slower) approach: build final image pixel by pixel (or vectorized)
        # using the calculated alpha and indices
        # blurred_image = (1 - alpha) * blurred_layers[indices] + alpha * blurred_layers[indices + 1] # Needs broadcasting/indexing fix

        # More explicit loop (easier to understand, maybe slower):
        print(f"DEBUG Defocus: Blending {num_levels} blur levels...")
        final_blurred = np.zeros_like(image_data, dtype=np.float32)
        alpha = alpha.flatten() # Flatten alpha and indices for easier iteration if needed
        indices = indices.flatten()
        img_flat = image_data.flatten() # Not needed if accessing layers directly

        # Get flattened versions of the layer images
        layers_flat = [layer.flatten() for layer in blurred_layers]

        # Vectorized interpolation:
        layer_0_flat = np.take_along_axis(np.array(layers_flat).T, indices[:, np.newaxis], axis=1).squeeze()
        layer_1_flat = np.take_along_axis(np.array(layers_flat).T, (indices + 1)[:, np.newaxis], axis=1).squeeze()

        final_blurred_flat = (1.0 - alpha) * layer_0_flat + alpha * layer_1_flat

        final_blurred = final_blurred_flat.reshape(image_data.shape)
        print("DEBUG Defocus: Blending complete.")


        return final_blurred.astype(image_data.dtype)