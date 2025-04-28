import torch
from torch.utils.data import Dataset, Subset # Import Subset
import os
import glob
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random # For shuffling indices
from typing import List

class SEMSegmentationDataset(Dataset):
    def __init__(self, data_dir: str,
                 max_layers: int, # Add max_layers argument
                 indices: List[int] = None,
                 transform: A.Compose = None,
                 ignore_border_pixels: int = 0):
        """
        Args:
            data_dir (str): Path to the root directory containing sample subdirs.
            mask_type (str): Which mask file to load.
            indices (List[int]): List of indices to include in this dataset instance (for train/val/test). If None, use all.
            transform (A.Compose): Albumentations transform pipeline.
            ignore_border_pixels (int): Pixels to ignore at border.
        """
        super().__init__()
        self.max_layers = max_layers # Store max_layers
        self.data_dir = data_dir
        self.transform = transform
        self.ignore_border_pixels = ignore_border_pixels

        # Find all samples (top-level images define the samples)
        all_image_files = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
        if not all_image_files:
            all_image_files = sorted(glob.glob(os.path.join(data_dir, "*.png")))

        if not all_image_files:
            raise FileNotFoundError(f"No valid image files (.tif or .png) found in {data_dir}")

        # --- Use provided indices to select files for this split ---
        if indices is not None:
            self.image_files = [all_image_files[i] for i in indices if i < len(all_image_files)]
        else:
            self.image_files = all_image_files # Use all if no indices provided

        print(f"Initialized dataset split with {len(self.image_files)} samples.")
        if not self.image_files:
             print(f"Warning: No samples selected for this dataset split (Indices: {indices})")


    def __len__(self):
        return len(self.image_files)


    def __getitem__(self, idx):
        if idx >= len(self.image_files):
             raise IndexError("Index out of bounds for dataset split")
        image_path = self.image_files[idx]
        # Derive subdir name from image name (e.g., 'sem_00001.tif' -> 'sem_00001')
        # Load image
        subdir_name = os.path.splitext(os.path.basename(image_path))[0]
        subdir_path = os.path.join(self.data_dir, subdir_name)
        img_vis_path = os.path.join(subdir_path, "image_final_noisy.png")
        # ... (fallback logic as before) ...
        image = cv2.imread(img_vis_path, cv2.IMREAD_GRAYSCALE)
        if image is None: raise IOError(...)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=-1) # HWC for Albumentations

        # --- Load and Create Multi-Class Layer Mask ---
        height, width = image.shape[:2]
        # Initialize target mask with background class (0)
        layer_target_mask = np.zeros((height, width), dtype=np.int64) # Use int64 for LongTensor

        # Find layer mask files for this sample
        subdir_name = os.path.splitext(os.path.basename(self.image_files[idx]))[0]
        subdir_path = os.path.join(self.data_dir, subdir_name)
        masks_data_dir = os.path.join(subdir_path, "layers") # Dir containing layer_xx dirs

        # Load individual actual layer masks (npy recommended)
        # Layer order matters for overlaps: later layers overwrite earlier ones
        num_layers_found = 0
        for layer_idx in range(self.max_layers): # Iterate up to max possible layers
             layer_dir = os.path.join(masks_data_dir, f"layer_{layer_idx:02d}")
             mask_path = os.path.join(layer_dir, "actual_mask.npy") # Load data mask
             if os.path.exists(mask_path):
                 try:
                     layer_mask = np.load(mask_path) # uint8 0/1 mask
                     if layer_mask.shape == (height, width):
                          # Assign layer class ID (layer 0 -> class 1, layer 1 -> class 2, ...)
                          layer_class_id = layer_idx + 1
                          layer_target_mask[layer_mask > 0] = layer_class_id
                          num_layers_found += 1
                     else:
                          print(f"Warning: Mask shape mismatch for {mask_path}. Skipping.")
                 except Exception as e:
                     print(f"Warning: Failed to load layer mask {mask_path}: {e}")
             else:
                  # Stop looking if a layer is missing sequentially? Or allow gaps?
                  # Assuming sequential for now, break if layer dir/mask missing.
                  # If layers can be sparse (e.g., layer 0, layer 2 exist but not 1),
                  # this needs adjustment based on how layers are stored/named.
                  # print(f"DEBUG: Layer {layer_idx} mask not found, stopping layer load.")
                  # break # If layers must be sequential
                  pass # Allow sparse layers if naming permits


        if num_layers_found == 0:
             print(f"Warning: No layer masks found for sample {subdir_name}. Target mask is all background.")

        # --- Create Valid Mask (as before) ---
        valid_mask = np.ones(layer_target_mask.shape, dtype=np.float32)
        b = self.ignore_border_pixels
        if b > 0:
            valid_mask[:b, :] = 0.0; valid_mask[-b:, :] = 0.0
            valid_mask[:, :b] = 0.0; valid_mask[:, -b:] = 0.0

        # --- Apply Albumentations Transforms ---
        sample = {'image': image, 'mask': layer_target_mask, 'valid_mask': valid_mask}
        if self.transform:
            # IMPORTANT: Use nearest neighbor interpolation for the multi-class mask!
            # Ensure your Albumentations Resize/ShiftScaleRotate use interpolation=cv2.INTER_NEAREST for masks.
            # Albumentations usually does this correctly if mask is passed to 'mask'.
            augmented = self.transform(**sample)
            sample = augmented

        # --- Convert to Tensor ---
        image = sample['image'] # Should be (C, H, W) float from ToTensorV2
        mask = sample['mask']   # Should be (H, W) int64 from ToTensorV2 (no channel dim for target)
        valid_mask = sample['valid_mask'] # Should be (H, W) float from ToTensorV2
             
        if not isinstance(mask, torch.LongTensor): mask = mask.long() # Ensure Long
        if mask.ndim != 2 or mask.shape[0] != image.shape[1] or mask.shape[1] != image.shape[2]: # Check shape is (H, W)
            print(f"ERROR: Final mask shape is {mask.shape}, expected ({image.shape[1]}, {image.shape[2]})")
            # Handle error: maybe return None or raise? Or try to fix?

        return {"image": image, "mask": mask, "valid_mask": valid_mask} # Return processed tensors