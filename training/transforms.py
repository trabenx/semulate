import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_train_transforms(height: int, width: int) -> A.Compose:
    return A.Compose([
        # --- Geometric ---
        # Resize first if fixing size
        A.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # Add slight rotation/scale/shift? Generator already does strong warps.
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10,
                           interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=0.3),

        # --- Color / Intensity ---
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        # A.CLAHE(p=0.2), # Can sometimes enhance contrast

        # --- Blur / Noise ---
        A.GaussNoise(var_limit=(10.0 / 255.0, 40.0 / 255.0), p=0.3), # Add extra noise
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),

        # --- Normalization / Tensor ---
        # Normalization might not be needed if input is already [0, 1]
        # A.Normalize(mean=(0.5,), std=(0.5,)), # Example if needed
        ToTensorV2(), # Converts numpy HWC -> torch CHW, scales int [0,255] to [0,1] if needed
    ])

def get_val_test_transforms(height: int, width: int) -> A.Compose:
    # Only resize and convert to tensor for validation/testing
    return A.Compose([
        A.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR, always_apply=True),
        # A.Normalize(mean=(0.5,), std=(0.5,)), # Apply if used in training
        ToTensorV2(),
    ])