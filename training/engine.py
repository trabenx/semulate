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
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    # Use SMP metrics - expects logits/probabilities and integer target masks
    # For binary case (sigmoid output):
    dice_metric = smp_metrics.f1_score(threshold=0.5).to(device) # F1 is Dice for binary

    pbar = tqdm(dataloader, desc="Validation", leave=False)

    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device) # Expecting (N, 1, H, W) float for loss, need int for metric
        valid_masks = batch['valid_mask'].to(device)

        with torch.cuda.amp.autocast(enabled=device=='cuda'):
             outputs = model(images) # Logits (N, 1, H, W)
             loss = criterion(outputs, masks, valid_masks) # Loss uses logits and float masks

        total_loss += loss.item()

        # --- Calculate Metrics ---
        # Need probabilities (apply sigmoid) and integer target mask
        preds_prob = torch.sigmoid(outputs)
        # Target mask needs to be integer type (0 or 1) and match spatial dims
        masks_int = (masks > 0.5).long() # Convert float [0,1] mask to long [0,1]
        # Apply valid mask to predictions and targets before metric calculation? Optional.
        # Example: preds_prob = preds_prob * valid_mask; masks_int = masks_int * valid_mask.long()
        dice_metric.update(preds_prob, masks_int)

    avg_loss = total_loss / len(dataloader)
    final_dice = dice_metric.compute().item() # Compute final metric and get scalar value

    print(f"Validation Dice: {final_dice:.4f}")

    return avg_loss, final_dice # Return both loss and metric