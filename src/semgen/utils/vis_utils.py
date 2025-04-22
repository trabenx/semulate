# src/semgen/utils/vis_utils.py
from typing import Optional
import numpy as np
import cv2 # Need OpenCV for color space conversion

def visualize_warp_field(
    warp_field: np.ndarray,
    max_expected_disp: Optional[float] = None,
    scaling_factor: float = 1.0
) -> np.ndarray:
    """
    Visualizes a 2D displacement field (dx, dy) using HSV color space.

    Direction is mapped to Hue, Magnitude is mapped to Value/Saturation.

    Args:
        warp_field (np.ndarray): The displacement field array with shape (H, W, 2),
                                 where warp_field[..., 0] is dx and warp_field[..., 1] is dy.
        max_expected_disp (Optional[float]): The maximum displacement magnitude expected.
                                             Used to normalize the magnitude for brightness/saturation.
                                             If None, it's calculated from the input field's max magnitude.
        scaling_factor (float): An optional factor to amplify the displacement magnitudes
                                before normalization, useful if displacements are very small.

    Returns:
        np.ndarray: An 8-bit BGR image visualizing the warp field.
    """
    if warp_field is None or warp_field.shape[2] != 2:
        raise ValueError("Input warp_field must be a HxWx2 NumPy array.")

    rows, cols = warp_field.shape[:2]
    dx = warp_field[..., 0] * scaling_factor
    dy = warp_field[..., 1] * scaling_factor

    # --- Calculate Magnitude and Angle ---
    magnitude = np.sqrt(dx**2 + dy**2)
    # Angle in radians, range [-pi, pi]
    angle_rad = np.arctan2(dy, dx)
    # Convert angle to degrees [0, 360] for Hue mapping
    angle_deg = np.degrees(angle_rad)
    angle_deg[angle_deg < 0] += 360 # Map negative angles to 180-360 range

    # --- Map to HSV ---
    # Hue: Map angle [0, 360] to [0, 179] (OpenCV HSV range)
    hue = (angle_deg / 2.0).astype(np.uint8)

    # Saturation: Use constant full saturation for clarity
    saturation = np.full((rows, cols), 255, dtype=np.uint8)

    # Value (Brightness): Map magnitude to brightness
    if max_expected_disp is None:
        # Auto-scale based on max magnitude in the current field
        max_mag = np.max(magnitude)
        if max_mag < 1e-6: # Avoid division by zero if field is all zeros
            print("DEBUG vis_warp: Warp field magnitude is near zero.")
            value = np.zeros((rows, cols), dtype=np.uint8) # Black image
        else:
            norm_magnitude = magnitude / max_mag
            value = (np.clip(norm_magnitude, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        # Scale based on the provided expected maximum displacement
        if max_expected_disp <= 0:
            print("Warning vis_warp: max_expected_disp should be positive. Using auto-scaling.")
            max_mag = np.max(magnitude)
            if max_mag < 1e-6: value = np.zeros((rows, cols), dtype=np.uint8)
            else: value = (np.clip(magnitude / max_mag, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            norm_magnitude = magnitude / max_expected_disp
            value = (np.clip(norm_magnitude, 0.0, 1.0) * 255).astype(np.uint8)


    # --- Combine HSV channels and convert to BGR ---
    hsv_image = cv2.merge([hue, saturation, value])
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return bgr_image