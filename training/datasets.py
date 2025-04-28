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
        self.mask_type = mask_type
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

        # Load mask
        mask = None
        # Construct mask path based on mask_type
        if self.mask_type == "instance_mask":
            # Load uint16 TIF recommended
            mask_path = os.path.join(subdir_path, "instance_mask.tif")
            if os.path.exists(mask_path):
                 mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # Load as is (uint16)
                 if mask is not None and mask.ndim == 3: mask = mask[..., 0] # Handle potential extra channel read by cv2
            else: print(f"Warning: Instance mask not found: {mask_path}")

        elif self.mask_type == "combined_actual_mask":
             # Load NPY data mask (binary 0/1)
             mask_path = os.path.join(subdir_path, "combined_actual_mask.npy")
             if os.path.exists(mask_path):
                  mask = np.load(mask_path) # Loads as uint8 (0/1)
             else: print(f"Warning: Combined actual mask (.npy) not found: {mask_path}")
             # Fallback to loading visual PNG if NPY missing?
             if mask is None:
                   mask_vis_path = os.path.join(subdir_path, "combined_actual_mask_vis.png")
                   if os.path.exists(mask_vis_path):
                       mask_vis = cv2.imread(mask_vis_path, cv2.IMREAD_GRAYSCALE)
                       if mask_vis is not None: mask = (mask_vis > 128).astype(np.uint8) # Convert 0/255 back to 0/1

        if mask is None: mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Create valid mask
        valid_mask = np.ones(mask.shape[:2], dtype=np.float32) # Use H, W from mask
        b = self.ignore_border_pixels
        if b > 0 and mask.ndim == 2: # Ensure mask is 2D for slicing
            valid_mask[:b, :] = 0.0; valid_mask[-b:, :] = 0.0
            valid_mask[:, :b] = 0.0; valid_mask[:, -b:] = 0.0

        # Apply transforms
        sample = {'image': image, 'mask': mask, 'valid_mask': valid_mask}
        if self.transform:
            # Important: Ensure mask interpolation is NEAREST if doing geometric transforms!
            # Albumentations usually handles this if mask is passed to 'mask' arg.
            augmented = self.transform(**sample)
            sample = augmented

        # Convert to tensor (handled by ToTensorV2 in transforms)
        image = sample['image']
        mask = sample['mask']
        valid_mask = sample['valid_mask']

        # Final mask shape/type adjustment
        if self.mask_type == "combined_actual_mask":
             mask = mask.float() # Keep (1, H, W) or (H, W) depending on ToTensorV2 output and loss
             if mask.ndim == 2: mask = mask.unsqueeze(0) # Ensure channel dim -> (1, H, W) for BCE/Dice
        elif self.mask_type == "instance_mask":
             mask = mask.long() # Keep (H, W) LongTensor
             
        if not isinstance(mask, torch.LongTensor): mask = mask.long() # Ensure Long
        if mask.ndim != 2 or mask.shape[0] != image.shape[1] or mask.shape[1] != image.shape[2]: # Check shape is (H, W)
            print(f"ERROR: Final mask shape is {mask.shape}, expected ({image.shape[1]}, {image.shape[2]})")
            # Handle error: maybe return None or raise? Or try to fix?

        print(f"DEBUG Dataset: Returning Image Shape: {image.shape}, Mask Shape: {mask.shape}, Mask Dtype: {mask.dtype}")

        return {"image": image, "mask": mask, "valid_mask": valid_mask} # Return processed tensors