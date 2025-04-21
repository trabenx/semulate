import numpy as np
from typing import List

def generate_cumulative_mask(layer_masks: List[np.ndarray]) -> List[np.ndarray]:
    """
    Generates cumulative masks from a list of individual layer masks.

    Args:
        layer_masks (List[np.ndarray]): List of binary (uint8) masks, one per layer,
                                       in the order they are applied.

    Returns:
        List[np.ndarray]: List where element i is the combination of masks 0 to i.
    """
    if not layer_masks:
        return []

    cumulative_masks = []
    current_cumulative = np.zeros_like(layer_masks[0], dtype=np.uint8)

    for mask in layer_masks:
        if mask is not None:
            # Combine using logical OR (or maximum if non-binary masks were possible)
            current_cumulative = np.logical_or(current_cumulative, mask).astype(np.uint8)
        # Append a copy so later modifications don't affect previous results
        cumulative_masks.append(current_cumulative.copy())

    return cumulative_masks


def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
    """
    Combines a list of masks into a single composite mask.

    Args:
        masks (List[np.ndarray]): List of binary (uint8) masks.

    Returns:
        np.ndarray: A single binary mask representing the union of input masks.
    """
    if not masks:
        return None # Or return an empty mask?

    # Find the first non-None mask to get shape and dtype
    first_valid_mask = next((m for m in masks if m is not None), None)
    if first_valid_mask is None:
        return None # All masks were None

    combined = np.zeros_like(first_valid_mask, dtype=np.uint8)
    for mask in masks:
        if mask is not None:
            combined = np.logical_or(combined, mask).astype(np.uint8)

    return combined