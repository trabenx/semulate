# inference/predict.py
import torch
import os
import glob
import yaml
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import time
import json

# Need to add src to path to import model builder etc.
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Add project root
if src_path not in sys.path:
    sys.path.insert(0, src_path)     # Add src path

# Use absolute imports now src is in path
from semgen.utils import save_image, save_npy, save_metadata, ensure_dir_exists # From generator utils
from training.models.unet import build_model # Assuming relative path is okay now or add training to path
from training.utils import load_checkpoint # Import checkpoint loader
from utils_inference import (preprocess_image, postprocess_prediction,
                             create_overlay, create_contour_overlay) # Import inference utils

def predict(args):
    """Runs inference on input images using a trained model."""
    print("--- Starting Inference ---")

    # --- Load Training Config ---
    print(f"Loading training config from: {args.train_config}")
    if not os.path.exists(args.train_config):
        raise FileNotFoundError(f"Training config file not found: {args.train_config}")
    with open(args.train_config, 'r') as f:
        config = yaml.safe_load(f)

    cfg_model = config['model']
    cfg_aug = config['augmentation']

    # Determine model input size (if fixed during training)
    target_size = None
    if cfg_aug.get('resize_height') and cfg_aug.get('resize_width'):
        target_size = (cfg_aug['resize_height'], cfg_aug['resize_width'])
        print(f"Model expects input size (H, W): {target_size}")
    else:
        print("Warning: Model trained without fixed resize? Inference might be suboptimal or fail if architecture requires fixed size.")
        # Some models might handle variable input (e.g., fully convolutional without GAP)

    num_classes = cfg_model['classes']
    is_binary = num_classes == 1
    print(f"Model configured for {num_classes} classes {'(Binary)' if is_binary else '(Multi-class)'}.")


    # --- Setup Device ---
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA specified but not available. Using CPU.")
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    print("Loading model...")
    model = build_model(
        arch=cfg_model['arch'],
        encoder_name=cfg_model['encoder_name'],
        encoder_weights=None, # Don't need pretrained weights for inference usually
        in_channels=cfg_model['in_channels'],
        classes=cfg_model['classes']
    )
    print(f"Loading checkpoint from: {args.checkpoint}")
    epoch = load_checkpoint(args.checkpoint, model) # Load weights onto CPU first
    if epoch is None:
         print(f"ERROR: Failed to load checkpoint {args.checkpoint}")
         return
    model.to(device)
    model.eval() # Set model to evaluation mode

    # --- Prepare Output Directory ---
    ensure_dir_exists(args.output_dir)
    print(f"Saving outputs to: {args.output_dir}")

    # --- Find Input Images ---
    image_paths = []
    for ext in args.extensions.split(','):
         pattern = os.path.join(args.input_dir, f"*.{ext.strip()}")
         image_paths.extend(glob.glob(pattern))

    if not image_paths:
        print(f"Error: No images found in {args.input_dir} with extensions {args.extensions}")
        return

    print(f"Found {len(image_paths)} images to process.")

    # --- Inference Loop ---
    inference_times = []
    for img_path in tqdm(image_paths, desc="Processing Images"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        try:
            start_time = time.time()

            # 1. Preprocess Image
            # Returns input tensor (N,C,H,W), original BGR image (H,W,C), original size (H,W)
            input_tensor, original_img_bgr, original_size = preprocess_image(img_path, target_size)
            input_tensor = input_tensor.to(device)

            # 2. Run Model Prediction
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(device == 'cuda')): # Use AMP if available
                    output_logits = model(input_tensor)

            # 3. Postprocess Prediction
            # Returns class map (H_orig, W_orig) uint8, prob map (H_orig, W_orig) float
            class_map, prob_map = postprocess_prediction(
                output_logits.cpu(), # Move output to CPU for processing
                original_size,
                target_size,
                num_classes,
                is_binary
            )
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # --- 4. Save Outputs ---
            output_subdir = os.path.join(args.output_dir, base_name)
            ensure_dir_exists(output_subdir)

            # Save Raw Prediction (e.g., class map)
            save_image(class_map, os.path.join(output_subdir, f"{base_name}_pred_mask.png")) # Simple 0/1 or 0..N image
            if args.save_npy: save_npy(class_map, os.path.join(output_subdir, f"{base_name}_pred_mask.npy"))
            if args.save_prob_map: save_image((prob_map * 255).astype(np.uint8), os.path.join(output_subdir, f"{base_name}_pred_prob.png"))
            if args.save_prob_map and args.save_npy: save_npy(prob_map, os.path.join(output_subdir, f"{base_name}_pred_prob.npy"))


            # Save Color Overlay
            overlay_img = create_overlay(original_img_bgr, class_map, alpha=args.overlay_alpha)
            save_image(overlay_img, os.path.join(output_subdir, f"{base_name}_overlay.png"))

            # Save Contour Overlay
            contour_img = create_contour_overlay(original_img_bgr, class_map)
            save_image(contour_img, os.path.join(output_subdir, f"{base_name}_contours.png"))

            # Save One-Hot Masks (if multiclass)
            if not is_binary and args.save_one_hot:
                one_hot_dir = os.path.join(output_subdir, "one_hot_masks")
                ensure_dir_exists(one_hot_dir)
                for class_id in range(num_classes): # Include background? Usually skip background (class 0)
                     if class_id == 0 and not args.save_background_one_hot: continue
                     one_hot_mask = (class_map == class_id).astype(np.uint8) * 255
                     save_image(one_hot_mask, os.path.join(one_hot_dir, f"{base_name}_class_{class_id:02d}.png"))

            # Save simple metadata
            meta = {
                 "source_image": img_path,
                 "model_checkpoint": args.checkpoint,
                 "original_size": original_size,
                 "processed_size": target_size if target_size else original_size,
                 "inference_time_sec": round(inference_time, 4),
                 "num_classes_predicted": int(np.max(class_map)) # Max class ID found
            }
            save_metadata(meta, os.path.join(output_subdir, f"{base_name}_inference_meta.json"))


        except Exception as e:
            print(f"\nError processing {base_name}: {e}")
            import traceback
            traceback.print_exc() # Print traceback for debugging


    # --- Print Summary ---
    if inference_times:
        avg_time = np.mean(inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        print("\n--- Inference Summary ---")
        print(f"Processed {len(inference_times)} images.")
        print(f"Average inference time: {avg_time:.4f} seconds")
        print(f"Average FPS: {fps:.2f}")
        print(f"Outputs saved in: {os.path.abspath(args.output_dir)}")
        print("-------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run segmentation inference on SEM images.")
    parser.add_argument("-i", "--input-dir", required=True, help="Directory containing input images.")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory to save prediction outputs.")
    parser.add_argument("-c", "--checkpoint", required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("-t", "--train-config", required=True, help="Path to the training configuration YAML file used for the model.")
    parser.add_argument("-d", "--device", default="cuda", help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--extensions", default="png,tif,tiff,jpg,jpeg", help="Comma-separated list of image extensions to process.")
    parser.add_argument("--overlay-alpha", type=float, default=0.4, help="Transparency alpha for the colored overlay.")
    parser.add_argument("--save-npy", action="store_true", help="Save raw prediction mask/probability map as .npy file.")
    parser.add_argument("--save-prob-map", action="store_true", help="Save probability map visualization.")
    parser.add_argument("--save-one-hot", action="store_true", help="Save individual binary masks for each predicted class (multiclass only).")
    parser.add_argument("--save-background-one-hot", action="store_true", help="Include class 0 (background) when saving one-hot masks.")

    args = parser.parse_args()
    predict(args)