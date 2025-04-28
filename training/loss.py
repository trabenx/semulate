import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch.losses as smp_losses

class SegmentationLoss(nn.Module):
    def __init__(self, loss_type: str = 'ce', num_classes: int = 2, ignore_border_pixels: int = 0):

        super().__init__()
        self.num_classes = num_classes # Ensure this is passed correctly from config
        self.ignore_border_pixels = ignore_border_pixels
        self.loss_type = loss_type.lower()

        # Use activation within loss for binary case (sigmoid)
        if self.loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        elif self.loss_type == 'dice':
             self.criterion = smp_losses.DiceLoss(mode='multiclass', from_logits=True, smooth=1e-6) # Set mode='multiclass'
        elif self.loss_type == 'ce_dice':
             self.ce = nn.CrossEntropyLoss(reduction='none')
             self.dice = smp_losses.DiceLoss(mode='multiclass', from_logits=True, smooth=1e-6) # Set mode='multiclass'
             # Define how to combine them in forward
             self.criterion = lambda p, t: (self.ce(p,t), self.dice(p,t)) # Return both components
        elif self.loss_type == 'iou' or self.loss_type == 'jaccard':
             self.criterion = smp_losses.JaccardLoss(mode='multiclass', from_logits=True) # Set mode='multiclass'
        elif self.loss_type == 'focal':
             self.criterion = smp_losses.FocalLoss(mode='multiclass', gamma=2.0) # Set mode='multiclass'
        else:
             raise ValueError(f"Unsupported loss type for multiclass: {loss_type}")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # Expected Input Shapes:
        # y_pred: (N, C, H, W) - Float Logits
        # y_true: (N, H, W) - Long Indices
        # valid_mask: (N, H, W) - Float Mask (0 or 1)

        # --- Ensure valid_mask has same spatial dims (N, H, W) ---
        if valid_mask.ndim == 4: # If accidentally (N, 1, H, W)
            valid_mask = valid_mask.squeeze(1)
        if valid_mask.shape != y_true.shape: # Should be (N, H, W) == (N, H, W)
            raise ValueError(f"Shape mismatch: valid_mask {valid_mask.shape}, y_true {y_true.shape}")

        # --- Calculate raw loss per pixel ---
        if self.loss_type == 'ce_dice':
            # self.criterion returns a TUPLE: (ce_loss_map, dice_loss_scalar)
            ce_loss_map, dice_loss_scalar = self.criterion(y_pred, y_true)
            # ce_loss_map shape: (N, H, W) - Float per-pixel CE loss
            # dice_loss_scalar shape: torch.Size([]) - Scalar Float Dice loss

            # Apply valid mask to CE map
            # *** Check shapes here ***
            # masked_loss = loss_map * valid_mask # Original problematic line?
            masked_ce_loss = ce_loss_map * valid_mask # Use the correct variable! Shape (N, H, W) * (N, H, W) -> OK

            num_valid_pixels = torch.sum(valid_mask) + 1e-8
            mean_ce_loss = torch.sum(masked_ce_loss) / num_valid_pixels

            # Combine the averaged CE loss and the scalar Dice loss
            mean_loss = 0.5 * mean_ce_loss + 0.5 * dice_loss_scalar

        # --- Handling for other single-component losses ---
        elif self.loss_type in ['ce', 'focal']: # Losses returning a map (N, H, W)
            loss_map = self.criterion(y_pred, y_true) # Should be (N, H, W)
            if loss_map.shape != y_true.shape: # Add shape check
                 raise RuntimeError(f"Loss map shape {loss_map.shape} != target shape {y_true.shape} for loss {self.loss_type}")
            masked_loss = loss_map * valid_mask # Element-wise multiplication
            num_valid_pixels = torch.sum(valid_mask) + 1e-8
            mean_loss = torch.sum(masked_loss) / num_valid_pixels

        elif self.loss_type in ['dice', 'iou', 'jaccard']: # Losses returning a scalar
            loss_scalar = self.criterion(y_pred, y_true) # Should be scalar (shape [])
            if loss_scalar.ndim != 0: # Add shape check
                 raise RuntimeError(f"Expected scalar loss for {self.loss_type}, got shape {loss_scalar.shape}")
            # No masking needed for scalar loss (implicitly calculated over whole batch/image by SMP)
            mean_loss = loss_scalar
        else:
            raise ValueError(f"Loss calculation not defined for type: {self.loss_type}")

        return mean_loss
