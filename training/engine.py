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
    dice_metric = smp_metrics.f1_score(num_classes=num_classes if num_classes > 1 else None, # Specify num_classes if > 1 for clarity? SMP might infer. Let's rely on inference for now.
                                       ignore_index=None).to(device) # Can ignore background index if needed
    iou_metric = smp_metrics.iou_score(num_classes=num_classes if num_classes > 1 else None,
                                       ignore_index=None).to(device)
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
    if num_classes > 1:
        mean_dice = torch.mean(final_dice_per_class[1:]).item() # Avg Dice over foreground classes
        mean_iou = torch.mean(final_iou_per_class[1:]).item()   # Avg IoU over foreground classes
    else: # Binary case (num_classes should technically be 1)
         mean_dice = final_dice_per_class[0].item() # Only one class value
         mean_iou = final_iou_per_class[0].item()

    print(f"Validation Dice: {mean_dice:.4f} | Validation IoU: {mean_iou:.4f}")

    return avg_loss, mean_dice # Return loss and primary metric (e.g., Dice)