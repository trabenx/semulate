# inference/utils_inference.py
import numpy as np
import cv2
import torch
from typing import List, Tuple, Optional
import random # For distinct colors

# Define colors (can share with generator's vis_utils if refactored)
DISTINCT_COLORS_BGR_INF = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (255, 128, 0),   # Orange-ish Blue
    (128, 0, 255),   # Purple
    (0, 255, 128),   # Teal-ish Green
    (128, 128, 128), # Gray
    (192, 192, 192), # Silver
    (128, 128, 0),   # Olive
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Teal
    (255, 165, 0),   # Orange
    (0, 128, 0),     # Dark Green
    (128, 0, 0),     # Navy
    # Add more colors if expecting > 17 layers
]
# Generate more random colors if needed
# random.seed(42)
# for _ in range(50):
#     DISTINCT_COLORS_BGR_INF.append(tuple(random.randint(50, 255) for _ in range(3)))


def preprocess_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    """Loads, preprocesses image for model input (resize, normalize, tensor)."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError(f"Cannot load image: {image_path}")

    original_size = (image.shape[0], image.shape[1]) # H, W
    original_image_display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # Keep original for overlay

    # --- Resize if target_size is provided ---
    if target_size is not None:
        height, width = target_size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        height, width = original_size # Use original size

    # --- Normalize and Convert to Tensor ---
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=-1) # Add channel dim -> H, W, C
    image = torch.from_numpy(image).permute(2, 0, 1).float() # -> C, H, W tensor
    image = image.unsqueeze(0) # Add batch dim -> N, C, H, W (N=1)

    return image, original_image_display, original_size

def postprocess_prediction(
    prediction: torch.Tensor, # Raw model output logits (N, C, H, W) or probs
    original_size: Tuple[int, int], # H, W
    target_size: Optional[Tuple[int, int]], # H, W (size model saw)
    num_classes: int,
    is_binary: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Converts model output tensor to class prediction map and probability map."""
    prediction = prediction.squeeze(0) # Remove batch dim -> (C, H, W)

    # --- Resize back to original size if needed ---
    # Use NEAREST neighbor for class predictions, maybe linear for probabilities?
    current_size = (prediction.shape[1], prediction.shape[2])
    if target_size is not None and target_size != original_size:
        # Resize predicted class map using NEAREST
        # Need class map first (argmax) BEFORE resizing
        pass # Handle resizing after getting class map / probs

    # --- Get Class Predictions ---
    if is_binary: # Binary segmentation (C=1)
        probs = torch.sigmoid(prediction) # (1, H, W)
        class_map = (probs > 0.5).squeeze(0).cpu().numpy().astype(np.uint8) # (H, W) binary map 0 or 1
        prob_map = probs.squeeze(0).cpu().numpy() # (H, W) probability map [0,1]
    else: # Multi-class (C > 1)
        probs = torch.softmax(prediction, dim=0) # (C, H, W) probabilities per class
        class_map = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8) # (H, W) class indices
        # Probability map could be max probability, or specific class prob? Let's use max prob.
        prob_map, _ = torch.max(probs, dim=0)
        prob_map = prob_map.cpu().numpy() # (H, W) max probability map

    # --- Resize maps back to original image size ---
    if target_size is not None and target_size != original_size:
        orig_h, orig_w = original_size
        # Resize class map using NEAREST
        class_map = cv2.resize(class_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        # Resize probability map using LINEAR
        prob_map = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return class_map, prob_map # Return uint8 class map and float probability map


def create_overlay(original_image: np.ndarray, class_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Creates a semi-transparent colored overlay on the original image."""
    if original_image.ndim == 2: # Convert original to color if it's grayscale
        overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = original_image.copy()

    if class_map is None: return overlay

    colored_mask = np.zeros_like(overlay, dtype=np.uint8)
    num_colors = len(DISTINCT_COLORS_BGR_INF)

    # Skip background class 0
    for class_id in range(1, np.max(class_map) + 1):
        color = DISTINCT_COLORS_BGR_INF[class_id % num_colors]
        colored_mask[class_map == class_id] = color

    # Blend overlay
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay


def create_contour_overlay(original_image: np.ndarray, class_map: np.ndarray) -> np.ndarray:
    """Draws contours of segmented regions onto the original image."""
    if original_image.ndim == 2:
        contour_img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        contour_img = original_image.copy()

    if class_map is None: return contour_img

    num_colors = len(DISTINCT_COLORS_BGR_INF)
    # Find contours for each class separately
    for class_id in range(1, np.max(class_map) + 1):
        binary_mask = np.uint8(class_map == class_id)
        if np.sum(binary_mask) == 0: continue # Skip if class not present

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = DISTINCT_COLORS_BGR_INF[class_id % num_colors]
        cv2.drawContours(contour_img, contours, -1, color, thickness=1) # Draw thin contours

    return contour_img