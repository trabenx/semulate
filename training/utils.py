import torch
import os
import shutil

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Saves model checkpoint."""
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Save temporarily then move to ensure atomicity
    temp_filepath = filepath + ".tmp"
    torch.save(state, temp_filepath)
    shutil.move(temp_filepath, filepath) # Atomic move/rename
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
     """Loads model checkpoint."""
     if not os.path.exists(filepath):
         print(f"Warning: Checkpoint '{filepath}' not found.")
         return None
     checkpoint = torch.load(filepath, map_location='cpu') # Load to CPU first
     model.load_state_dict(checkpoint['state_dict'])
     if optimizer and 'optimizer' in checkpoint:
         optimizer.load_state_dict(checkpoint['optimizer'])
     print(f"Checkpoint loaded from '{filepath}' (Epoch {checkpoint.get('epoch', -1)}, Loss {checkpoint.get('loss', float('inf')):.4f})")
     return checkpoint.get('epoch', 0)

# Add metric calculation functions (IoU, Dice) here if not using a library
def calculate_iou(preds, targets, smooth=1e-6):
    # preds, targets expected to be binary (0 or 1), shape (N, H, W) or (N, 1, H, W)
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    total = (preds + targets).sum()
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou