import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch.losses as smp_losses

class SegmentationLoss(nn.Module):
    def __init__(self, loss_type: str = 'dice_bce', ignore_border_pixels: int = 0):
        super().__init__()
        self.ignore_border_pixels = ignore_border_pixels
        self.loss_type = loss_type.lower()

        # Use activation within loss for binary case (sigmoid)
        if self.loss_type == 'dice':
            self.criterion = smp_losses.DiceLoss(mode='binary', from_logits=True)
        elif self.loss_type == 'bce':
             # BCEWithLogitsLoss includes sigmoid
            self.criterion = nn.BCEWithLogitsLoss(reduction='none') # No reduction yet
        elif self.loss_type == 'dice_bce':
             self.dice = smp_losses.DiceLoss(mode='binary', from_logits=True, smooth=1e-6)
             self.bce = nn.BCEWithLogitsLoss(reduction='none')
             self.criterion = lambda p, t: 0.5 * self.dice(p, t) + 0.5 * self.bce(p, t)
        elif self.loss_type == 'iou' or self.loss_type == 'jaccard':
             self.criterion = smp_losses.JaccardLoss(mode='binary', from_logits=True)
        elif self.loss_type == 'focal':
             self.criterion = smp_losses.FocalLoss(mode='binary', gamma=2.0) # Gamma default 2.0
        else:
             raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates loss, ignoring borders based on valid_mask.

        Args:
            y_pred (torch.Tensor): Model predictions (logits), shape (N, C, H, W). C=1 for binary.
            y_true (torch.Tensor): Ground truth masks, shape (N, C, H, W) float for binary BCE/Dice.
            valid_mask (torch.Tensor): Mask indicating valid pixels (1=valid, 0=ignore), shape (N, H, W) or (N, 1, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # --- Ensure valid_mask has same spatial dims and potentially channel dim ---
        if valid_mask.ndim == 3: # N, H, W -> add channel dim if needed
            valid_mask = valid_mask.unsqueeze(1) # N, 1, H, W
        if valid_mask.shape[1] != y_pred.shape[1]: # Check channel dim match (esp. C=1)
             valid_mask = valid_mask.expand_as(y_pred) # Expand C dim if necessary

        # --- Calculate raw loss per pixel ---
        loss_map = self.criterion(y_pred, y_true) # Shape (N, C, H, W) usually

        # --- Apply valid mask and calculate mean ---
        # Multiply loss element-wise by the valid mask
        masked_loss = loss_map * valid_mask

        # Calculate mean loss only over valid pixels
        # Sum the masked loss and divide by the number of valid pixels
        num_valid_pixels = torch.sum(valid_mask) + 1e-8 # Add epsilon for stability
        mean_loss = torch.sum(masked_loss) / num_valid_pixels

        return mean_loss