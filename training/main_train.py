import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from datasets import SEMSegmentationDataset

import yaml
import os
import random
import numpy as np
import time
import glob 
from datasets import SEMSegmentationDataset
from transforms import get_train_transforms, get_val_test_transforms
from models.unet import build_model # Assuming unet.py contains build_model
from loss import SegmentationLoss
from engine import train_one_epoch, evaluate
from utils import save_checkpoint # Need a utils.py with save_checkpoint

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(config_path='training/config_train.yaml'):
    # --- Load Config ---
    print(f"Loading training configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    cfg_data = config['data']
    cfg_model = config['model']
    cfg_train = config['training']
    cfg_aug = config['augmentation']
    cfg_log = config['logging']

    # --- Setup ---
    set_seed(config.get('seed', 42))
    device = torch.device(cfg_train['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directories
    os.makedirs(cfg_log['log_dir'], exist_ok=True)
    os.makedirs(cfg_log['checkpoint_dir'], exist_ok=True)
    writer = SummaryWriter(log_dir=cfg_log['log_dir'])

    # --- Data ---
    print("Setting up datasets and dataloaders...")
    resize_h = cfg_aug.get('resize_height')
    resize_w = cfg_aug.get('resize_width')
    if not resize_h or not resize_w:
        raise ValueError("resize_height and resize_width must be specified in augmentation config.")

    train_tfms = get_train_transforms(resize_h, resize_w)
    val_tfms = get_val_test_transforms(resize_h, resize_w)

    # --- Implement Index-based Split ---
    # 1. Find all sample indices first
    all_files = sorted(glob.glob(os.path.join(cfg_data['synthetic_data_dir'], "*.tif")))
    if not all_files: all_files = sorted(glob.glob(os.path.join(cfg_data['synthetic_data_dir'], "*.png")))
    num_all_samples = len(all_files)
    all_indices = list(range(num_all_samples))
    rng_split = random.Random(config.get('seed', 42)) # Use seeded RNG for split
    rng_split.shuffle(all_indices) # Shuffle indices in place

    # 2. Calculate split sizes
    train_size = int(cfg_data['train_split'] * num_all_samples)
    val_size = int(cfg_data['val_split'] * num_all_samples)
    test_size = num_all_samples - train_size - val_size

    if train_size == 0 or val_size == 0:
         raise ValueError("Train or validation split size is zero. Adjust splits or check dataset size.")

    # 3. Get indices for each split
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size : train_size + val_size]
    test_indices = all_indices[train_size + val_size :]

    print(f"Total samples: {num_all_samples}. Splitting indices: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    # 4. Create Datasets with specific indices and transforms
    dataset = SEMSegmentationDataset(
        data_dir=cfg_data['synthetic_data_dir'],
        max_layers=cfg_model['max_layers'], # Pass max_layers
        ignore_border_pixels=cfg_data['ignore_border_pixels']
    )
    
    train_dataset = Subset(dataset, train_indices) # Use Subset or handle indices/transforms differently
    val_dataset = Subset(dataset, val_indices)

    train_dataset.dataset.transform = train_tfms
    val_dataset.dataset.transform = val_tfms
    # test_dataset = SEMSegmentationDataset(...) # Create if needed

    # --- End Index-based Split ---

    train_loader = DataLoader(train_dataset, batch_size=cfg_train['batch_size'], shuffle=True, num_workers=cfg_data['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg_train['batch_size'] * 2, shuffle=False, num_workers=cfg_data['num_workers'], pin_memory=True)
    # test_loader = DataLoader(test_dataset, ...) # For final evaluation

    # --- Model ---
    print("Building model...")
    model = build_model(
        arch=cfg_model['arch'],
        encoder_name=cfg_model['encoder_name'],
        encoder_weights=cfg_model['encoder_weights'],
        in_channels=cfg_model['in_channels'],
        classes=cfg_model['classes']
    )
    model.to(device)

    # --- Loss, Optimizer, Scheduler ---
    print("Setting up loss, optimizer, scheduler...")
    criterion = SegmentationLoss(
        loss_type=cfg_train['loss'],
        num_classes=cfg_model['classes'],
        ignore_border_pixels=cfg_data['ignore_border_pixels'])

    if cfg_train['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg_train['learning_rate'], weight_decay=cfg_train['weight_decay'])
    elif cfg_train['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg_train['learning_rate'], weight_decay=cfg_train['weight_decay'])
    else: # Add SGD etc.
        raise ValueError(f"Unsupported optimizer: {cfg_train['optimizer']}")

    scheduler = None
    sched_params = cfg_train.get('lr_scheduler_params', {})
    if cfg_train['lr_scheduler'] == 'step':
        scheduler = StepLR(optimizer, step_size=sched_params.get('step_size', 10), gamma=0.1)
    elif cfg_train['lr_scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=sched_params.get('T_max', cfg_train['epochs']), eta_min=1e-6)
    elif cfg_train['lr_scheduler'] == 'reduce_lr_on_plateau':
         scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=sched_params.get('patience', 5), verbose=True)

    # AMP Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=cfg_train['mixed_precision'] and device=='cuda')

    # --- Training Loop ---
    print("Starting training loop...")
    best_val_loss = float('inf')

    for epoch in range(1, cfg_train['epochs'] + 1):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch}/{cfg_train['epochs']} ---")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, cfg_train['gradient_clipping'])
        print(f"Epoch {epoch} Train Loss: {train_loss:.4f}")
        writer.add_scalar('Loss/train', train_loss, epoch)

        # Validation
        if epoch % cfg_log['val_freq'] == 0:
            # Pass val_loader to evaluate
            val_loss, val_dice = evaluate(model, val_loader, criterion, device) # Capture metrics
            print(f"Epoch {epoch} Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}") # Print metrics
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Dice/val', val_dice, epoch) # Log metrics

            # LR Scheduler Step (for ReduceLROnPlateau)
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)

            # Save Best Model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(cfg_log['checkpoint_dir'], f"model_best.pth")
                save_checkpoint(model, optimizer, epoch, val_loss, save_path)
                print(f"Best model saved to {save_path}")

        # LR Scheduler Step (for StepLR, CosineAnnealingLR)
        if scheduler and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()

        # Save Checkpoint periodically
        if epoch % cfg_log['save_freq'] == 0:
             save_path = os.path.join(cfg_log['checkpoint_dir'], f"model_epoch_{epoch}.pth")
             save_checkpoint(model, optimizer, epoch, val_loss if 'val_loss' in locals() else -1, save_path) # Save last val_loss


        # Log LR
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch duration: {epoch_duration:.2f}s")


    print("Training finished.")
    writer.close()

if __name__ == '__main__':
    # Add argument parsing to specify config file if desired
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', default='config_train.yaml')
    # args = parser.parse_args()
    # main(args.config)
    main() # Runs with default config path