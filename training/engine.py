import torch
from tqdm import tqdm
import gc # Garbage collection
import segmentation_models_pytorch.metrics as smp_metrics # Import metrics


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
    
    # --- Initialize metrics for MULTICLASS ---
    # Remove threshold argument, it's handled internally via argmax for multiclass
    dice_metric = smp_metrics.f1_score(ignore_index=None).to(device)
    iou_metric = smp_metrics.iou_score(ignore_index=None).to(device)
    # --- End Metric Initialization Change ---

    pbar = tqdm(dataloader, desc="Validation", leave=False)

    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device) # Expecting (N, 1, H, W) float for loss, need int for metric
        valid_masks = batch['valid_mask'].to(device)

        with torch.cuda.amp.autocast(enabled=device=='cuda'):
             outputs = model(images) # Logits (N, 1, H, W)
             loss = criterion(outputs, masks, valid_masks) # Loss uses logits and float masks

        total_loss += loss.item()

        # --- Update Metrics ---
        # SMP metrics generally expect logits for multiclass when threshold=None
        # Pass raw logits (N, C, H, W) and integer target masks (N, H, W)
        # The metric applies argmax internally
        # Note: If using ignore_index, ensure background class ID is correct
        dice_metric.update(outputs, masks)
        iou_metric.update(outputs, masks)

    avg_loss = total_loss / len(dataloader)
    # Compute final metrics
    final_dice_per_class = dice_metric.compute() # Returns tensor (C,)
    final_iou_per_class = iou_metric.compute()   # Returns tensor (C,)

    # Calculate mean metrics (e.g., excluding background class 0)
    if num_classes > 1 and len(final_dice_per_class) > 1: # Check length defensively
        # Average over classes 1 to C-1
        mean_dice = torch.mean(final_dice_per_class[1:]).item()
        mean_iou = torch.mean(final_iou_per_class[1:]).item()
    elif len(final_dice_per_class) > 0: # Handle binary case or if only background predicted?
         mean_dice = final_dice_per_class[0].item() # Metric for the only class (or background)
         mean_iou = final_iou_per_class[0].item()
    else: # Should not happen if dataloader has items
        mean_dice = 0.0
        mean_iou = 0.0

    print(f"Validation Dice: {mean_dice:.4f} | Validation IoU: {mean_iou:.4f}")

    return avg_loss, mean_dice # Return loss and primary metric (e.g., Dice)