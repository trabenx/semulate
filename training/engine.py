import torch
import numpy as np
from tqdm import tqdm
import gc # Garbage collection
import segmentation_models_pytorch.losses as smp_losses


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None, max_grad_norm=1.0):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        valid_masks = batch['valid_mask'].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, masks, valid_masks)

        if scaler:
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                 scaler.unscale_(optimizer) # Unscale gradients before clipping
                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Clear CUDA cache periodically if memory is tight
        # if i % 50 == 0:
        #    torch.cuda.empty_cache()
        #    gc.collect()


    return epoch_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    
    # --- Use Loss instances to calculate batch metrics ---
    # Initialize Dice and IoU Loss objects configured for multiclass evaluation
    # We want the metric value (1 - loss for Dice/IoU usually, but SMP might return score directly)
    # Use high epsilon to avoid division by zero issues in eval if a class isn't present
    dice_metric_calculator = smp_losses.DiceLoss(mode='multiclass', from_logits=True, smooth=1e-5)
    iou_metric_calculator = smp_losses.JaccardLoss(mode='multiclass', from_logits=True, smooth=1e-5) # Jaccard is IoU

    # Accumulators for metrics over the epoch
    epoch_dice_scores = []
    epoch_iou_scores = []

    pbar = tqdm(dataloader, desc="Validation", leave=False)


    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device) # Expecting (N, 1, H, W) float for loss, need int for metric
        valid_masks = batch['valid_mask'].to(device)

        with torch.cuda.amp.autocast(enabled=device=='cuda'):
             outputs = model(images) # Logits (N, 1, H, W)
             loss = criterion(outputs, masks, valid_masks) # Loss uses logits and float masks

        total_loss += loss.item()

        # --- Calculate Batch Metrics using Loss functions ---
        # NOTE: These loss functions calculate the loss *per image* in the batch then average.
        # They might not directly give per-class scores easily without modification or using different tools.
        # Let's calculate the batch-averaged Dice and IoU scores (which are 1 - Loss).

        # Calculate Dice score = 1 - DiceLoss
        # Need to apply valid_mask *before* calculating metric if possible,
        # but SMP losses might not support mask input directly.
        # Alternative: Calculate metric on full image, accept it's less precise due to borders.
        batch_dice_loss = dice_metric_calculator(outputs, masks)
        batch_dice_score = 1.0 - batch_dice_loss.item() # Get scalar score for the batch
        epoch_dice_scores.append(batch_dice_score)

        # Calculate IoU score = 1 - JaccardLoss
        batch_iou_loss = iou_metric_calculator(outputs, masks)
        batch_iou_score = 1.0 - batch_iou_loss.item() # Get scalar score for the batch
        epoch_iou_scores.append(batch_iou_score)

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_dice_score:.4f}", iou=f"{batch_iou_score:.4f}")


    avg_loss = total_loss / len(dataloader)
    # --- Calculate Epoch Mean Metrics ---
    final_mean_dice = np.mean(epoch_dice_scores) if epoch_dice_scores else 0.0
    final_mean_iou = np.mean(epoch_iou_scores) if epoch_iou_scores else 0.0


    print(f"Validation Mean Dice: {final_mean_dice:.4f} | Validation Mean IoU: {final_mean_iou:.4f}")
    # Note: These are average scores across batches, potentially including background influence
    # depending on how the SMP losses calculate the multiclass average.


    return avg_loss, final_mean_dice # Return loss and primary metric (e.g., Dice)