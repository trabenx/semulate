from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

@dataclass
class GeneratedSample:
    """Holds all components of a generated sample before saving."""
    sample_id: int
    output_dir: str

    # --- Images ---
    image_final: Optional[np.ndarray] = None       # The main output image
    image_clean: Optional[np.ndarray] = None       # Image before global noise/intensity artifacts
    background: Optional[np.ndarray] = None      # Initial background

    # --- Masks ---
    # Layer Masks (List per layer: [original_mask, actual_mask])
    layer_masks: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = field(default_factory=list)
    cumulative_masks: List[Optional[np.ndarray]] = field(default_factory=list) # Cumulative actual masks
    combined_original_mask: Optional[np.ndarray] = None
    combined_actual_mask: Optional[np.ndarray] = None
    instance_mask: Optional[np.ndarray] = None     # Optional instance segmentation mask
    defect_mask: Optional[np.ndarray] = None       # Optional mask of structural defects
    noise_mask: Optional[np.ndarray] = None        # Optional noise map

    # --- Metadata & Debug ---
    metadata: Dict[str, Any] = field(default_factory=dict) # Comprehensive generation info
    overlay_image: Optional[np.ndarray] = None     # Optional debug overlay render
    metadata_overlay_image: Optional[np.ndarray] = None # Scale bar / text overlay render
    warp_field: Optional[np.ndarray] = None        # Optional displacement field (H, W, 2)

    # --- File Paths (Populated during saving) ---
    output_paths: Dict[str, str] = field(default_factory=dict)

    def add_layer_masks(self, original: Optional[np.ndarray], actual: Optional[np.ndarray]):
        self.layer_masks.append((original, actual))

    def get_actual_masks(self) -> List[np.ndarray]:
        """Returns a list of only the 'actual' masks for layers."""
        return [actual for _, actual in self.layer_masks if actual is not None]

    def get_original_masks(self) -> List[np.ndarray]:
        """Returns a list of only the 'original' masks for layers."""
        return [original for original, _ in self.layer_masks if original is not None]