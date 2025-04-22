import numpy as np
import cv2
import os
import time
import random
import logging
import hashlib
from typing import Dict, Any, Optional, List, Tuple

from .core import GeneratedSample
from .config_loader import load_config # Assuming using this loader
from .raffle import Raffler
from .shapes import create_shape
from .artifacts import (create_artifact, AffineTransform, ElasticMeshDeform, TopographicShading)
from .noise import create_noise
from .utils import (generate_procedural_noise_2d, save_gif, visualize_warp_field,
                   save_image, save_metadata, save_text, ensure_dir_exists,
                   generate_cumulative_mask, combine_masks, scale_to_uint)
from .artifacts.shape_level import EdgeRipple, BreaksHoles

# Potentially import background generation functions if they become complex
# from .background import generate_background

# Placeholder for background generation
def generate_background(config: Dict[str, Any], size: Tuple[int, int], rng: random.Random) -> np.ndarray:
    bg_type = config.get('background_type', 'flat')
    height, width = size
    # Default to float32 [0,1] for internal processing
    background = np.zeros((height, width), dtype=np.float32)
    
    # --- Define helper to generate a single component ---
    # This avoids duplicating gradient/noise logic
    def generate_component(comp_type: str, comp_config: Dict[str, Any]) -> np.ndarray:
        comp_background = np.zeros(size, dtype=np.float32)
        if comp_type == 'flat':
            intensity = float(comp_config.get('flat_intensity', 0.1))
            comp_background.fill(intensity)
        elif comp_type == 'gradient':
            params = comp_config.get('gradient_params', {})
            start = float(params.get('start_intensity', 0.0))
            end = float(params.get('end_intensity', 0.2))
            style = comp_config.get('gradient_style', 'linear')
            if style == 'linear':
                direction_deg = float(params.get('direction', 0.0))  # Angle in degrees
                direction_rad = np.radians(direction_deg)
                # Project coordinates onto the gradient direction vector
                y_coords, x_coords = np.mgrid[0:height, 0:width]
                # Center coords for rotation pivot? Or project from corner? Project from 0,0 is simpler.
                proj_coords = x_coords * np.cos(direction_rad) + y_coords * np.sin(direction_rad)
                # Normalize projected coords to roughly [0, 1] across the image dimension in that direction
                max_proj = width * abs(np.cos(direction_rad)) + height * abs(np.sin(direction_rad))
                norm_proj = proj_coords / max_proj if max_proj > 0 else np.zeros_like(proj_coords, dtype=float)  # Avoid division by zero if angle lines up poorly
                comp_background = start + (end - start) * np.clip(norm_proj, 0.0, 1.0) # Linear interpolation
            elif style == 'radial':
                center_x_rel = float(params.get('center_x%', 0.5))
                center_y_rel = float(params.get('center_y%', 0.5))
                center_x = center_x_rel * width
                center_y = center_y_rel * height
                # Calculate distance of each pixel from the center
                y_coords, x_coords = np.mgrid[0:height, 0:width]
                distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                # Normalize distance (e.g., by max distance to corner, or half diagonal)
                max_dist = np.sqrt((width/2.0)**2 + (height/2.0)**2)
                norm_dist = distance / max_dist if max_dist > 0 else np.zeros_like(distance, dtype=float)
                # Interpolate intensity based on normalized distance
                comp_background = start + (end - start) * np.clip(norm_dist, 0.0, 1.0)
            else:
                print(f"Warning: Unknown gradient style '{style}'. Using flat average.")
                comp_background.fill((start + end)/2)
        elif comp_type == 'noise':
            noise_amplitude = float(comp_config.get('noise_amplitude', 0.05))
            noise_type = comp_config.get('noise_type', 'perlin')
            noise_base = 0.0 # For component noise, base is often 0, add later
            scale_param = float(comp_config.get('noise_frequency', 10.0))
            scale = width / max(1.0, scale_param)
            octaves = int(comp_config.get('noise_octaves', 4))
            persistence = float(comp_config.get('noise_persistence', 0.5))
            lacunarity = float(comp_config.get('noise_lacunarity', 2.0))
            base_seed = rng.randint(0, 100000)
            if noise_type in ['perlin', 'simplex']:
                # Generate noise centered at 0 for easier combination [-amp/2, +amp/2] approx
                noise_map = generate_procedural_noise_2d(
                    shape=size, scale=scale, octaves=octaves, persistence=persistence,
                    lacunarity=lacunarity, base_seed=base_seed, noise_type=noise_type,
                    normalize_range=(-0.5, 0.5) # Range centered at 0
                )
                comp_background = noise_map * noise_amplitude # Scale centered noise
            elif noise_type == 'gaussian':
                 comp_background = np.random.normal(loc=0.0, scale=noise_amplitude, size=size)
            else: # Fallback
                comp_background.fill(0.0)
        return comp_background.astype(np.float32)
    # --- End Helper ---
    
    if bg_type == 'flat':
        background = generate_component('flat', config)
    elif bg_type == 'gradient':
        # Pass only gradient relevant parts of config
        grad_conf = {'gradient_params': config.get('gradient_params'),
                     'gradient_style': config.get('gradient_style')}
        background = generate_component('gradient', grad_conf)
    elif bg_type == 'noise':
         # Pass only noise relevant parts of config
         noise_conf = {k: v for k, v in config.items() if k.startswith('noise_')}
         background = generate_component('noise', noise_conf)
         # Add base intensity AFTER generating noise centered at 0
         base_intensity = float(config.get('noise_base_intensity', 0.05))
         background += base_intensity

    elif bg_type == 'composite':
        print("Generating composite background...")
        # Define components and combination method (make configurable?)
        comp1_type = config.get('composite_comp1_type', 'gradient') # e.g., gradient
        comp2_type = config.get('composite_comp2_type', 'noise')    # e.g., noise
        combine_mode = config.get('composite_combine_mode', 'add') # 'add', 'multiply', 'overlay'

        # Generate component 1 (e.g., gradient)
        # Need to extract relevant params, potentially prefixed in config
        comp1_config = config.get('composite_comp1_params', {})
        # If params aren't prefixed, pull from top level based on type
        if not comp1_config:
             if comp1_type == 'gradient': comp1_config = {'gradient_params': config.get('gradient_params'), 'gradient_style': config.get('gradient_style')}
             elif comp1_type == 'noise': comp1_config = {k: v for k, v in config.items() if k.startswith('noise_')}
             elif comp1_type == 'flat': comp1_config = {'flat_intensity': config.get('flat_intensity')}

        component1 = generate_component(comp1_type, comp1_config)

        # Generate component 2 (e.g., noise)
        comp2_config = config.get('composite_comp2_params', {})
        if not comp2_config:
             if comp2_type == 'gradient': comp2_config = {'gradient_params': config.get('gradient_params'), 'gradient_style': config.get('gradient_style')}
             elif comp2_type == 'noise': comp2_config = {k: v for k, v in config.items() if k.startswith('noise_')}
             elif comp2_type == 'flat': comp2_config = {'flat_intensity': config.get('flat_intensity')}

        component2 = generate_component(comp2_type, comp2_config)

        # Combine components
        print(f"Combining '{comp1_type}' and '{comp2_type}' using mode '{combine_mode}'")
        if combine_mode == 'add':
             # Add components, maybe add a base offset too?
             base_intensity = float(config.get('composite_base_intensity', 0.05))
             background = base_intensity + component1 + component2 # Add noise/gradient deviations to base
        elif combine_mode == 'multiply':
             # Often base * (1 + noise) * gradient_factor
             # Assuming component1=gradient [0,1], component2=noise [-a/2, a/2]
             base_intensity = float(config.get('composite_base_intensity', 0.1))
             background = base_intensity * component1 * (1.0 + component2) # Example combination
        # TODO: Implement other modes like 'overlay' if needed
        else: # Default to add
             print(f"Warning: Unknown composite combine mode '{combine_mode}'. Using 'add'.")
             base_intensity = float(config.get('composite_base_intensity', 0.05))
             background = base_intensity + component1 + component2

        # Clip final composite background
        np.clip(background, 0.0, 1.0, out=background)

    else: # Unknown type
        print(f"Warning: Unknown background type '{bg_type}', using flat default.")
        background.fill(0.1)

    return background.astype(np.float32)

# Placeholder for overlay generation
def generate_debug_overlay(final_image: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    # Simple overlay: convert final to color, draw mask outlines
    if final_image.ndim == 2:
        overlay = cv2.cvtColor((final_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else: # Assume already color
        overlay = final_image.copy()

    colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)] # Green, Blue, Yellow, Cyan
    for i, mask in enumerate(masks):
        if mask is not None and np.max(mask) > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, colors[i % len(colors)], 1)
    return overlay

# Placeholder for metadata overlay
def generate_metadata_overlay(overlay_config: Dict[str, Any], global_config: Dict[str, Any], size: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Generates the metadata overlay image with scale bar and text.

    Args:
        overlay_config (Dict[str, Any]): The specific 'metadata_overlay' part of the config.
        global_config (Dict[str, Any]): The main configuration dict (needed for pixel_size_nm).
        size (Tuple[int, int]): The (height, width) of the image.

    Returns:
        Optional[np.ndarray]: The overlay image (uint8 BGR) or None if disabled.
    """
    if not overlay_config.get('enabled', True): # Default to True if key missing? Or False? Let's assume True if section exists.
        return None

    height, width = size
    # Create a black background (or maybe transparent if saving PNG?)
    # For simplicity, using black BGR background.
    meta_overlay = np.zeros((height, width, 3), dtype=np.uint8)

    # --- Get Parameters ---
    pixel_size_nm = global_config.get('pixel_size_nm')
    if pixel_size_nm is None:
        print("Warning: 'pixel_size_nm' not found in global config. Cannot draw scale bar accurately.")
        # Optionally draw text only? Or return None? Skipping scale bar for now.
        pixel_size_nm = 0 # Indicate error

    scale_bar_conf = overlay_config.get('scale_bar', {})
    text_info_conf = overlay_config.get('text_info', {})
    styling_conf = overlay_config.get('styling', {})

    # Styling
    try: # Handle hex colors if provided (basic implementation)
        font_color_str = styling_conf.get('font_color', 'white').lower()
        bar_color_str = styling_conf.get('bar_color', 'white').lower()
        # Simple color name mapping (add more or use a library)
        color_map = {'white': (255, 255, 255), 'black': (0, 0, 0), 'gray': (128, 128, 128), 'red': (0, 0, 255)}
        font_color = color_map.get(font_color_str, (255, 255, 255)) # BGR format
        bar_color = color_map.get(bar_color_str, (255, 255, 255))
    except Exception:
         font_color = (255, 255, 255)
         bar_color = (255, 255, 255)

    font_size_pt = int(text_info_conf.get('font_size_pt', 10))
    # OpenCV font scale is approximate from point size
    font_scale = font_size_pt / 20.0 # Adjust this factor based on visual results
    font_thickness = 1 # Can make configurable
    font_face = cv2.FONT_HERSHEY_SIMPLEX

    anchor = styling_conf.get('anchor', 'bottom_right')
    offset_x, offset_y = styling_conf.get('offset_px', [10, 10])
    margin = 5 # Internal margin between elements

    # --- Draw Scale Bar ---
    bar_drawn_height = 0
    text_height_estimate = 0 # Estimate height occupied by text later
    bar_length_real = float(scale_bar_conf.get('length', 1.0))
    bar_units = scale_bar_conf.get('units', 'µm')
    bar_thickness_px = int(scale_bar_conf.get('thickness_px', 4))

    if pixel_size_nm > 0:
        # Convert real length to pixels
        length_nm = bar_length_real
        if bar_units == 'µm': length_nm *= 1000
        elif bar_units == 'mm': length_nm *= 1000000
        bar_length_px = int(round(length_nm / pixel_size_nm))

        # Create label text (e.g., "1 µm")
        scale_label = f"{bar_length_real:.1f}".rstrip('0').rstrip('.') + f" {bar_units}"
        (label_w, label_h), baseline = cv2.getTextSize(scale_label, font_face, font_scale, font_thickness)
        text_height_estimate += label_h + baseline + margin
        bar_drawn_height = bar_thickness_px + margin + label_h + baseline

        # Calculate positions based on anchor
        # Bar bottom-left corner (bar_x, bar_y)
        # Label bottom-left corner (label_x, label_y)
        if anchor == 'bottom_right':
            bar_x = width - offset_x - bar_length_px
            bar_y = height - offset_y - bar_thickness_px - label_h - baseline - margin
            label_x = width - offset_x - max(bar_length_px // 2 + label_w // 2, label_w) # Center label approx
            label_y = height - offset_y - baseline
        elif anchor == 'bottom_left':
            bar_x = offset_x
            bar_y = height - offset_y - bar_thickness_px - label_h - baseline - margin
            label_x = offset_x + max(0, bar_length_px // 2 - label_w // 2) # Center label approx
            label_y = height - offset_y - baseline
        elif anchor == 'top_right':
             bar_x = width - offset_x - bar_length_px
             bar_y = offset_y + text_height_estimate # Place below text
             label_x = width - offset_x - max(bar_length_px // 2 + label_w // 2, label_w)
             label_y = offset_y + text_height_estimate + bar_thickness_px + label_h + margin
        elif anchor == 'top_left':
             bar_x = offset_x
             bar_y = offset_y + text_height_estimate # Place below text
             label_x = offset_x + max(0, bar_length_px // 2 - label_w // 2)
             label_y = offset_y + text_height_estimate + bar_thickness_px + label_h + margin
        else: # Default bottom right
             bar_x = width - offset_x - bar_length_px
             bar_y = height - offset_y - bar_thickness_px - label_h - baseline - margin
             label_x = width - offset_x - max(bar_length_px // 2 + label_w // 2, label_w)
             label_y = height - offset_y - baseline
             
        # print(f"DEBUG Overlay: pixel_size_nm={pixel_size_nm}")
        # print(f"DEBUG Overlay: bar_length_real={bar_length_real}, bar_units={bar_units}")
        # print(f"DEBUG Overlay: bar_length_px={bar_length_px}, bar_thickness_px={bar_thickness_px}")
        # print(f"DEBUG Overlay: label='{scale_label}', label_size=({label_w}, {label_h})")
        # print(f"DEBUG Overlay: bar_pos=({bar_x}, {bar_y}), label_pos=({label_x}, {label_y})")
        # print(f"DEBUG Overlay: bar_color={bar_color}, font_color={font_color}")
        # print(f"DEBUG Overlay: Image size=({width}, {height})")


        # Draw bar and label if bar fits
        if bar_length_px > 0 and bar_x >= 0 and (bar_x + bar_length_px) <= width and bar_y >= 0 and (bar_y + bar_thickness_px) <= height:
            # print(f"DEBUG Overlay: Drawing Bar Rect: ({bar_x}, {bar_y}) to ({bar_x + bar_length_px}, {bar_y + bar_thickness_px})") # Print draw coords
            cv2.rectangle(meta_overlay, (bar_x, bar_y), (bar_x + bar_length_px, bar_y + bar_thickness_px),
                          bar_color, thickness=-1)
            # print(f"DEBUG Overlay: Drawing Label Text at: ({label_x}, {label_y})") # Print draw coords
            cv2.putText(meta_overlay, scale_label, (label_x, label_y), font_face, font_scale,
                        font_color, font_thickness, lineType=cv2.LINE_AA)
        else:
             # This is important - why wasn't it drawn?
             print(f"DEBUG Overlay: Scale bar NOT drawn. Reason: Length={bar_length_px}, X Range=[{bar_x}, {bar_x+bar_length_px}], Y Range=[{bar_y}, {bar_y+bar_thickness_px}] vs Img Size=({width}, {height})")
             bar_drawn_height = 0

    # --- Draw Text Lines ---
    text_lines = text_info_conf.get('lines', [])
    current_text_y = 0 # Reset Y position calculation

    # Calculate text positions based on anchor
    if anchor.startswith('bottom'):
        # Start from bottom (above scale bar if drawn) and go up
        current_text_y = height - offset_y - bar_drawn_height - margin # Start above bar/offset
        for line in reversed(text_lines): # Draw from bottom up
            (text_w, text_h), baseline = cv2.getTextSize(line, font_face, font_scale, font_thickness)
            text_x = width - offset_x - text_w if anchor.endswith('right') else offset_x
            text_y = current_text_y - baseline # Position using baseline
            print(f"DEBUG Overlay: Drawing Text Line '{line}' at: ({text_x}, {text_y})")
            cv2.putText(meta_overlay, line, (text_x, text_y), font_face, font_scale,
                        font_color, font_thickness, lineType=cv2.LINE_AA)
            current_text_y -= (text_h + baseline + margin) # Move up for next line
    else: # top_left or top_right
        # Start from top and go down
        current_text_y = offset_y # Start below top offset
        for line in text_lines:
             (text_w, text_h), baseline = cv2.getTextSize(line, font_face, font_scale, font_thickness)
             text_x = width - offset_x - text_w if anchor.endswith('right') else offset_x
             # Position using text height + baseline for top alignment
             text_y = current_text_y + text_h
             cv2.putText(meta_overlay, line, (text_x, text_y), font_face, font_scale,
                         font_color, font_thickness, lineType=cv2.LINE_AA)
             current_text_y += (text_h + baseline + margin) # Move down for next line


    return meta_overlay

def generate_sample(config: Dict[str, Any], sample_index: int, base_output_dir: str, master_seed: int) -> GeneratedSample:
    """
    Generates a single synthetic SEM sample based on the configuration.

    Args:
        config (Dict[str, Any]): The full configuration dictionary.
        sample_index (int): The index number of this sample (e.g., 0, 1, ...).
        base_output_dir (str): The root directory where sample folders will be created.
        master_seed (int): The master random seed for the generator.

    Returns:
        GeneratedSample: A dataclass object containing the generated data.
    """
    # print(f"DEBUG: generate_sample received base_output_dir: '{base_output_dir}'") # Add this line
    start_time = time.time()
    output_opts = config.get('output_options', {}) # Define early
    save_logs_flag = output_opts.get('save_logs', False)
    
    # --- Setup Logger for this sample ---
    logger = logging.getLogger(f"sample_{sample_index}")
    logger.propagate = False # Prevent duplicate messages if root logger exists
    logger.setLevel(logging.DEBUG) # Log everything DEBUG and above
    # Remove existing handlers for this logger if re-running
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler (optional, good for seeing progress)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler) # Uncomment to see logs on console too
    
    # File handler (if enabled)
    file_handler = None
    log_file_path = None
    if save_logs_flag:
        sample_id_numeric = config.get('start_index', 0) + sample_index
        sample_id_str = f"sem_{sample_id_numeric:05d}"
        subdir_output_path_for_log = os.path.join(base_output_dir, sample_id_str) # Need path early
        logs_dir = os.path.join(subdir_output_path_for_log, "logs")
        ensure_dir_exists(logs_dir)
        log_file_path = os.path.join(logs_dir, "generation.log")

        file_handler = logging.FileHandler(log_file_path, mode='w') # Overwrite log each time
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
        logger.info(f"--- Starting Generation for Sample Index {sample_index} ---")
        logger.info(f"Using Master Seed: {master_seed}, Sample Seed: {master_seed + sample_index}")
    
    # --- 1. Initialization ---
    logger.info("Initialization...")
    file_suffix = f"_{sample_id_numeric:05d}"      # Suffix for main files
    
    # Define TWO output locations
    main_file_output_dir = base_output_dir # For image_final, meta.json etc.
    subdir_output_path = os.path.join(base_output_dir, sample_id_str) # For masks, debug etc.    
    
    ensure_dir_exists(main_file_output_dir) # Ensure base exists
    ensure_dir_exists(subdir_output_path)   # Ensure subdir exists
    ensure_dir_exists(os.path.join(subdir_output_path, "masks")) # Ensure masks subdir exists
    # ensure_dir_exists(os.path.join(subdir_output_path, "masks", "instance_masks")) # If needed

    # Seed RNG for this specific sample for reproducibility
    sample_seed = master_seed + sample_index
    rng = random.Random(sample_seed)
    np.random.seed(sample_seed) # Seed numpy's global RNG as well

    # Image dimensions
    width = config.get('image_width', 512)
    height = config.get('image_height', 512)
    size = (height, width)

    output_opts = config.get('output_options', {}) # Moved this line up
    mask_format = output_opts.get('output_formats', {}).get('masks', 'tif')
    
    # Create result holder
    sample = GeneratedSample(sample_id=sample_index, output_dir=subdir_output_path) # Store subdir as primary sample dir

    # --- 2. Raffle Effects ---
    logger.info("Raffling effects...")
    raffler = Raffler(config.get('artifact_raffle', {}), rng)
    applied_effects_list = raffler.raffle_effects_for_image() # List of {'type':..., 'parameters':..., 'order':...}
    logger.debug(f"Applied effects list: {applied_effects_list}")

    # --- 3. Background Generation ---
    logger.info("Generating background...")
    bg_config = config.get('background', {})
    # Pass logger to functions that might log warnings/errors? Optional.
    sample.background = generate_background(bg_config, size, rng) # Pass rng instance
    # Start with the background for clean image
    # Work with float32 internally
    current_image = sample.background.astype(np.float32).copy()

    # --- 4. Layer & Shape Generation ---
    logger.info("Starting layer/shape generation...")
    layering_config = config.get('layering', {})
    # num_layers and layer_defs are chosen/randomized by config_randomizer now
    num_layers = layering_config.get('num_layers', 0)
    layer_defs = layering_config.get('layers', [])

    all_shapes_in_sample_meta = [] # Store metadata for each shape instance
    all_shapes_final_actual_masks = [] # Store final masks for instance segmentation

    # Layer composition happens onto this buffer
    current_image = sample.background.astype(np.float32).copy()

    for layer_idx, layer_conf in enumerate(layer_defs):
        # This layer_conf is already randomized for specific values
        logger.info(f"-- Processing Layer {layer_idx}, Type: {layer_conf.get('shape_type')} --")
        if not layer_conf.get('enabled', True): continue # Should already be filtered, but double check

        pattern_conf = layer_conf.get('pattern', {})
        pattern_type = pattern_conf.get('type', 'single') # Default to single shape if no pattern
        shape_type = layer_conf.get('shape_type')
        if not shape_type:
            print(f"Error: Layer {layer_idx} missing 'shape_type'. Skipping.")
            continue

        # --- Generate Base Positions/Params based on Pattern ---
        shape_instance_configs = [] # List of config dicts for each shape in this layer
        base_shape_params = layer_conf.get('shape_params', {})

        if pattern_type == 'grid':
            rows = int(pattern_conf.get('rows', 3))
            cols = int(pattern_conf.get('cols', 3))
            spacing = float(pattern_conf.get('spacing', 50))
            jitter = float(pattern_conf.get('jitter_stddev', 0))
            offset_x, offset_y = pattern_conf.get('offset', [0,0])
            # TODO: Add grid orientation/skew logic if needed

            # Calculate grid start point (e.g., centered or corner)
            grid_width = (cols - 1) * spacing
            grid_height = (rows - 1) * spacing
            start_x = (width - grid_width) / 2.0 + offset_x
            start_y = (height - grid_height) / 2.0 + offset_y

            for r in range(rows):
                for c in range(cols):
                    jitter_x = np.random.normal(0, jitter) if jitter > 0 else 0
                    jitter_y = np.random.normal(0, jitter) if jitter > 0 else 0
                    cx = start_x + c * spacing + jitter_x
                    cy = start_y + r * spacing + jitter_y
                    shape_instance_configs.append({"center": (cx, cy)})

        elif pattern_type == 'hex_grid':
            # Hexagonal close packing grid
            spacing = float(pattern_conf.get('spacing', 50)) # Distance between centers
            rows = int(pattern_conf.get('rows', 5))
            cols = int(pattern_conf.get('cols', 5))
            jitter = float(pattern_conf.get('jitter_stddev', 0))
            offset_x, offset_y = pattern_conf.get('offset', [0,0])

            # Calculate grid start point (e.g., centered)
            hex_height = spacing * np.sqrt(3) / 2.0 # Vertical distance between rows
            grid_width = (cols - 1) * spacing
            grid_height = (rows - 1) * hex_height
            start_x = (width - grid_width) / 2.0 + offset_x
            start_y = (height - grid_height) / 2.0 + offset_y

            for r in range(rows):
                row_start_x = start_x + (r % 2) * (spacing / 2.0) # Offset every other row
                cy = start_y + r * hex_height + np.random.normal(0, jitter) if jitter > 0 else start_y + r * hex_height
                for c in range(cols if r % 2 == 0 else cols -1): # Adjust cols for staggered rows if needed, or keep rect bounding
                    cx = row_start_x + c * spacing + np.random.normal(0, jitter) if jitter > 0 else row_start_x + c * spacing
                    if 0 <= cx < width and 0 <= cy < height: # Basic bounds check
                        shape_instance_configs.append({"center": (cx, cy)})

        elif pattern_type == 'radial_grid':
            num_rings = int(pattern_conf.get('rings', 3))
            shapes_per_ring_base = int(pattern_conf.get('shapes_per_ring', 6))
            radius_step = float(pattern_conf.get('radius_step', 50)) # Distance between rings
            start_radius = float(pattern_conf.get('start_radius', radius_step)) # Radius of first ring
            center_x = float(pattern_conf.get('center_x%', 0.5)) * width # Relative center
            center_y = float(pattern_conf.get('center_y%', 0.5)) * height
            jitter_radius = float(pattern_conf.get('jitter_radius', 0))
            jitter_angle = float(pattern_conf.get('jitter_angle', 0)) # Degrees

            current_radius = start_radius
            for r in range(num_rings):
                # Increase shapes per ring slightly for outer rings?
                num_shapes_this_ring = shapes_per_ring_base + int(r * 1.5) # Example scaling
                if num_shapes_this_ring <= 0: continue

                actual_radius = current_radius + np.random.normal(0, jitter_radius) if jitter_radius > 0 else current_radius
                actual_radius = max(0, actual_radius)

                for i in range(num_shapes_this_ring):
                    base_angle = (i / num_shapes_this_ring) * 360.0 # Degrees
                    angle_offset = np.random.normal(0, jitter_angle) if jitter_angle > 0 else 0
                    actual_angle = base_angle + angle_offset
                    angle_rad = np.radians(actual_angle)

                    cx = center_x + actual_radius * np.cos(angle_rad)
                    cy = center_y + actual_radius * np.sin(angle_rad)

                    if 0 <= cx < width and 0 <= cy < height:
                        shape_instance_configs.append({"center": (cx, cy)})

                current_radius += radius_step

        elif pattern_type == 'random_scatter':
            num_shapes = int(layer_conf.get('num_shapes', 10))
            for _ in range(num_shapes):
                 cx = rng.uniform(0, width)
                 cy = rng.uniform(0, height)
                 shape_instance_configs.append({"center": (cx, cy)})

        elif pattern_type == 'full_span_vertical': # IMPLEMENTATION
            count = int(pattern_conf.get('count', 10))
            spacing_jitter_pct = float(pattern_conf.get('spacing_jitter%', 0.0))
            avg_spacing = width / float(count + 1)
            current_x = 0.0
            for i in range(count):
                spacing = avg_spacing * (1.0 + rng.uniform(-spacing_jitter_pct, spacing_jitter_pct)) if spacing_jitter_pct > 0 else avg_spacing
                current_x += spacing
                # Line needs center, length, angle
                cx = current_x
                cy = height / 2.0
                # Need to pass parameters expected by Line init OR calculate start/end here
                # Passing params for Line init:
                shape_instance_configs.append({
                    "center": (cx, cy),
                    "length": height, # Span full height
                    "rotation": 90.0 # Vertical angle
                 })

        elif pattern_type == 'full_span_horizontal': # IMPLEMENTATION
            count = int(pattern_conf.get('count', 10))
            spacing_jitter_pct = float(pattern_conf.get('spacing_jitter%', 0.0))
            avg_spacing = height / float(count + 1)
            current_y = 0.0
            for i in range(count):
                spacing = avg_spacing * (1.0 + rng.uniform(-spacing_jitter_pct, spacing_jitter_pct)) if spacing_jitter_pct > 0 else avg_spacing
                current_y += spacing
                shape_instance_configs.append({
                    "center": (width / 2.0, current_y),
                    "length": width, # Span full width
                    "rotation": 0.0 # Horizontal angle
                 })
        elif pattern_type == 'single': # Default if no pattern specified
             # Assume center is image center or randomize? Randomize for now.
             cx = rng.uniform(width * 0.2, width * 0.8)
             cy = rng.uniform(height * 0.2, height * 0.8)
             shape_instance_configs.append({"center": (cx, cy)})
        else:
             print(f"Warning: Unsupported pattern type '{pattern_type}' for layer {layer_idx}. Skipping.")
             continue

        # --- Process Each Shape Instance in the Layer ---
        layer_original_mask = np.zeros(size, dtype=np.uint8)
        layer_actual_mask = np.zeros(size, dtype=np.uint8)
        layer_shapes_meta = []

        for instance_idx, instance_conf in enumerate(shape_instance_configs):
            # Combine base params with instance-specific params (like center)
            # And potentially randomize other params per-instance if ranges were given
            concrete_shape_params = base_shape_params.copy()
            concrete_shape_params.update(instance_conf) # Add center etc.

            # --- Resolve Shape Type for this INSTANCE ---
            layer_shape_type_config = layer_conf.get('shape_type') # Could be string or list
            instance_shape_type = None
            if isinstance(layer_shape_type_config, list):
                if not layer_shape_type_config:
                    print(f"Error: Empty shape_type list for layer {layer_idx}. Skipping instance.")
                    continue
                instance_shape_type = rng.choice(layer_shape_type_config) # Choose one type for this instance
            elif isinstance(layer_shape_type_config, str):
                instance_shape_type = layer_shape_type_config # Use the single defined type
            else:
                print(f"Error: Invalid shape_type format for layer {layer_idx}: {layer_shape_type_config}. Skipping instance.")
                continue

            concrete_shape_params['shape_type'] = instance_shape_type # Set the chosen type for create_shape

            # Re-randomize per-instance if ranges exist in base_shape_params
            # (Requires careful use of config_randomizer helper or similar logic here)
            # Example: if isinstance(base_shape_params.get('rotation'), list):
            #              concrete_shape_params['rotation'] = raffler._randomize_parameter(base_shape_params['rotation'])
            # Example: if isinstance(base_shape_params.get('intensity'), list):
            #              concrete_shape_params['intensity'] = raffler._randomize_parameter(base_shape_params['intensity'])

            # Special handling for line parameters if needed (center already calculated)
            if shape_type == 'line':
                 if 'length' not in concrete_shape_params: concrete_shape_params['length'] = 50 # Default?
                 if 'thickness' not in concrete_shape_params: concrete_shape_params['thickness'] = 2
                 if 'rotation' not in concrete_shape_params: concrete_shape_params['rotation'] = rng.uniform(0, 180)
                 if 'intensity' not in concrete_shape_params: concrete_shape_params['intensity'] = 0.8


            # --- Create Shape Instance ---
            try:
                # Ensure intensity is resolved before creating shape
                if not isinstance(concrete_shape_params.get('intensity'), (int, float)):
                     concrete_shape_params['intensity'] = raffler._randomize_parameter(concrete_shape_params.get('intensity', [0.5, 0.8]))

                shape_instance = create_shape(concrete_shape_params)
                # Add to list if instance masks are needed later? Needs final mask though.
            except Exception as e:
                print(f"Error creating shape instance {instance_idx} (type: {instance_shape_type}) for layer {layer_idx}: {e}. Skipping instance.")
                continue

            # --- Generate Original Mask for this instance ---
            # Need to handle potential errors if shape is outside bounds
            try:
                 instance_original_mask = shape_instance.generate_mask(size)
                 logger.debug(f"Layer {layer_idx}, Instance {instance_idx}, Orig Mask Sum: {np.sum(instance_original_mask)}")
                 # Combine onto layer original mask
                 layer_original_mask = np.logical_or(layer_original_mask, instance_original_mask).astype(np.uint8)
            except Exception as e:
                 print(f"Error generating original mask for instance {instance_idx}, layer {layer_idx}: {e}")
                 instance_original_mask = np.zeros(size, dtype=np.uint8) # Use empty if error

            # --- Apply Shape-Level Artifacts (to this instance's mask) ---
            instance_actual_mask = instance_original_mask.copy()
            applied_shape_artifacts_meta = []
            # TODO: Decide how raffle applies - once per layer, or per shape? Assume per shape for now.
            for effect_info in applied_effects_list:
                 if effect_info['category'] == 'shape_level':
                     # Reroll probability per instance? Or apply if raffled for the image? Assume apply if raffled.
                     try:
                         artifact_instance = create_artifact(effect_info)
                         if isinstance(artifact_instance, (EdgeRipple, BreaksHoles)): # Modify mask
                             instance_actual_mask = artifact_instance.apply(instance_actual_mask)
                             applied_shape_artifacts_meta.append(effect_info['type']) # Just log type for now
                         # TODO: Handle vertex-modifying artifacts
                     except Exception as e:
                         print(f"Error applying shape artifact {effect_info['type']} to instance {instance_idx}: {e}")

            logger.debug(f"Layer {layer_idx}, Instance {instance_idx}, Actual Mask Sum: {np.sum(instance_actual_mask)}")
            # Combine onto layer actual mask
            layer_actual_mask = np.logical_or(layer_actual_mask, instance_actual_mask).astype(np.uint8)
            all_shapes_final_actual_masks.append(instance_actual_mask) # Store for instance seg

            # --- Store Metadata for this Instance ---
            shape_meta = {
                "instance_index": instance_idx,
                "type": shape_type,
                "parameters_used": concrete_shape_params, # Store final params
                "applied_shape_artifacts": applied_shape_artifacts_meta
            }
            layer_shapes_meta.append(shape_meta)

        # --- Finished processing shapes for this layer ---
        sample.add_layer_masks(layer_original_mask, layer_actual_mask) # Add combined masks for the layer
        all_shapes_in_sample_meta.append({ # Add layer meta info
            "layer_index": layer_idx,
            "shapes_generated": layer_shapes_meta
            # Store layer alpha, comp_mode etc if needed
        })

        # --- Compose Layer onto Main Image (Clean) ---
        alpha = float(layer_conf.get('alpha', 1.0))
        comp_mode = layer_conf.get('composition_mode', 'additive')
        # Intensity applied per shape now, how to handle layer composition?
        # Simplification: Use an average intensity or a fixed layer intensity?
        # Or: Render shapes with their intensity onto a temp layer buffer then composite buffer?

        # Let's use the layer_actual_mask and an average intensity for now
        layer_intensity_avg = np.mean([s['parameters_used']['intensity'] for s in layer_shapes_meta if 'intensity' in s['parameters_used']]) if layer_shapes_meta else 0.5

        if np.any(layer_actual_mask): # Only composite if mask isn't empty
            mask_pixels = layer_actual_mask > 0
            if comp_mode == 'additive':
                 current_image[mask_pixels] += layer_intensity_avg * alpha
            elif comp_mode == 'overwrite':
                 current_image[mask_pixels] = current_image[mask_pixels] * (1.0 - alpha) + layer_intensity_avg * alpha
            elif comp_mode == 'multiplicative':
                 current_image[mask_pixels] *= (1.0 - alpha * (1.0 - layer_intensity_avg))
            np.clip(current_image, 0.0, 1.0, out=current_image) # Clip after each layer

    sample.image_clean = current_image.copy() # Store state before global noise/intensity
    sample.metadata['layers'] = all_shapes_in_sample_meta # Store detailed shape metadata

    # --- 5. Generate Combined/Instance/Defect Masks ---
    actual_masks_list = sample.get_actual_masks() # These are now combined per layer
    original_masks_list = sample.get_original_masks()
    sample.combined_actual_mask = combine_masks(actual_masks_list)
    sample.combined_original_mask = combine_masks(original_masks_list)
    sample.cumulative_masks = generate_cumulative_mask(actual_masks_list)

    # --- Instance Mask Implementation ---
    if output_opts.get('generate_instance_masks', False):
        logger.info("Generating instance mask...")
        instance_mask_img = np.zeros(size, dtype=np.uint16) # Use uint16 for more IDs
        current_id = 1
        # Use the per-instance actual masks we stored
        for inst_mask in all_shapes_final_actual_masks:
            if inst_mask is not None and np.any(inst_mask):
                 instance_mask_img[inst_mask > 0] = current_id
                 current_id += 1
        sample.instance_mask = instance_mask_img
        logger.debug(f"Instance Mask Max ID: {np.max(instance_mask_img)}")
        if sample.instance_mask is not None:
             # Instance mask saving logic will be handled inside save_sample
             pass # No saving here, just generation
        # TODO: Save instance mask in save_sample
        ensure_dir_exists(os.path.join(sample.output_dir, "masks", "instance_masks")) # Redundant? Ensure it exists
        inst_fname = f"instance_mask.{mask_format}"
        inst_fpath = os.path.join(sample.output_dir, "masks", "instance_masks", inst_fname)
        # print(f"DEBUG: Saving instance mask to '{inst_fpath}'")
        save_image(sample.instance_mask, inst_fpath) # Save as uint16
        sample.output_paths['instance_mask'] = os.path.join(os.path.basename(sample.output_dir), "masks", "instance_masks", inst_fname) # Relative path


    # TODO: Implement defect mask generation more robustly

    # --- 6. Apply Image-Level Artifacts & Noise ---
    logger.info("Applying global effects...")
    final_image = sample.image_clean.copy()
    current_actual_masks = sample.get_actual_masks() # Keep track of masks needing geom warp

    for effect_info in applied_effects_list:
        category = effect_info['category']
        artifact_type = effect_info['type']
        parameters = effect_info['parameters'] # Get the parameters dict
        logger.debug(f"Applying global effect: {artifact_type} (Category: {category})")
        if category in ['image_level', 'instrument_effects', 'noise']:
            try:
                # Noise uses create_noise factory
                if category == 'noise':
                    # Option A: Modify create_noise to accept type, or
                    # Option B: Add type to params dict before passing
                    params_with_type = parameters.copy()
                    params_with_type['noise_type'] = artifact_type # Add type for factory
                    instance = create_noise(params_with_type) # Pass dict including type
                else: # Artifacts (image/instrument) use create_artifact
                    params_with_type = parameters.copy()
                    params_with_type['artifact_type'] = artifact_type # Consistency? Or just pass params.
                    # create_artifact expects {'type': ..., 'parameters': ...} structure from raffle output
                    instance = create_artifact(effect_info) # Pass the original effect_info dict
                    
                # --- Prepare kwargs for apply method ---
                apply_kwargs = {}
                if isinstance(instance, (AffineTransform, ElasticMeshDeform)):
                     apply_kwargs['masks'] = current_actual_masks
                elif isinstance(instance, TopographicShading): # Check specifically for TopoShading
                     # Pass cumulative masks if derived source is possible
                     if instance.get_param('height_map_source') == 'derived':
                          apply_kwargs['cumulative_masks'] = sample.cumulative_masks
                     # Pass RNG if perlin source is possible
                     if instance.get_param('height_map_source') == 'random_perlin':
                           apply_kwargs['rng'] = rng # Pass the sample's RNG

                # Apply based on type
                if isinstance(instance, (AffineTransform, ElasticMeshDeform)): # Geometric warps
                    logger.debug(f"Applying geometric warp '{artifact_type}'...")
                    # Default warp field is None
                    warp_field_generated = None

                    # Special handling for ElasticMeshDeform return value
                    if isinstance(instance, ElasticMeshDeform):
                        apply_kwargs['masks'] = current_actual_masks # Pass masks
                        # ElasticMeshDeform now returns 3 items
                        warped_image, warped_masks, warp_field_generated = instance.apply(final_image, **apply_kwargs)
                        if warp_field_generated is not None:
                             logger.debug("Captured warp field from ElasticMeshDeform.")
                             sample.warp_field = warp_field_generated # Store it in the sample object
                    elif isinstance(instance, AffineTransform):
                        apply_kwargs['masks'] = current_actual_masks # Pass masks
                        # Affine returns only 2 items
                        warped_image, warped_masks = instance.apply(final_image, **apply_kwargs)
                        # No warp field generated by affine in this implementation
                    
                    final_image = warped_image     # final_image remains an ndarray
                    current_actual_masks = warped_masks # Update the list of masks
                    logger.debug(f"Geometric warp '{artifact_type}' applied to main image and layer masks..")
                    
                    # --- Warp other masks (Combined, Instance, Defect) ---
                    # Define a helper list for masks to warp (handles None)
                    other_masks_to_warp = [
                        ('combined_actual', sample.combined_actual_mask),
                        ('instance', sample.instance_mask if output_opts.get('generate_instance_masks', False) else None),
                        ('defect', sample.defect_mask) # Assuming sample.defect_mask exists
                    ]
                    for mask_name, mask_obj in other_masks_to_warp:
                        if mask_obj is not None:
                            logger.debug(f"Warping '{mask_name}' mask...")
                            try:
                                # Call apply, passing the mask as the 'image' data and in the 'masks' list
                                warp_result = instance.apply(mask_obj, masks=[mask_obj])

                                # --- CORRECTED UNPACKING for other masks ---
                                # Check how many values were returned based on the instance type
                                if isinstance(instance, ElasticMeshDeform):
                                    # Expecting (warped_img, [warped_mask], warp_field)
                                    if len(warp_result) != 3:
                                         logger.error(f"ElasticMeshDeform apply for '{mask_name}' returned {len(warp_result)} values, expected 3!")
                                         continue # Skip updating this mask
                                    warped_mask_list = warp_result[1] # Get the list of masks (second element)
                                elif isinstance(instance, AffineTransform):
                                    # Expecting (warped_img, [warped_mask])
                                    if len(warp_result) != 2:
                                         logger.error(f"AffineTransform apply for '{mask_name}' returned {len(warp_result)} values, expected 2!")
                                         continue
                                    warped_mask_list = warp_result[1]
                                else: # Should not happen if logic above is correct
                                     logger.error("Unexpected instance type during auxiliary mask warp!")
                                     continue

                                # --- END CORRECTED UNPACKING ---

                                # Assign the warped mask back (assuming list contains one mask)
                                if warped_mask_list and warped_mask_list[0] is not None:
                                    if mask_name == 'combined_actual': sample.combined_actual_mask = warped_mask_list[0]
                                    elif mask_name == 'instance': sample.instance_mask = warped_mask_list[0]
                                    elif mask_name == 'defect': sample.defect_mask = warped_mask_list[0]
                                else:
                                     logger.warning(f"Warping '{mask_name}' mask resulted in None or empty list.")

                            except Exception as warp_err:
                                 logger.error(f"Error warping auxiliary mask '{mask_name}': {warp_err}")
                                 import traceback
                                 logger.error(traceback.format_exc())
                else: # Intensity, blur, noise artifacts
                    logger.debug(f"Applying intensity/noise effect '{artifact_type}'...")
                    # These return only the image, assignment is correct
                    # Ensure the instance.apply method ONLY returns the image ndarray
                    result = instance.apply(final_image, **apply_kwargs)
                    if isinstance(result, tuple):
                        logger.error(f"Intensity/Noise artifact '{artifact_type}' unexpectedly returned a tuple!")
                        # Handle error appropriately, maybe take first element?
                        final_image = result[0] # Risky assumption
                    else:
                         final_image = result
                    logger.debug(f"Effect '{artifact_type}' applied.")

            except Exception as e:
                 print(f"Error applying global effect {artifact_type}: {e}")
                 import traceback
                 traceback.print_exc() # Print full traceback for errors here

    # Now final_image should always be an ndarray when assigned below
    sample.image_final = final_image

    # --- 7. Generate Overlays ---
    if output_opts.get('save_debug_overlays', True): # Check default if needed
        logger.info("Generating overlays ...")
        sample.overlay_image = generate_debug_overlay(sample.image_final, current_actual_masks) # Use warped masks
    meta_overlay_conf = config.get('metadata_overlay', {})
    sample.metadata_overlay_image = generate_metadata_overlay(meta_overlay_conf, config, size)
        # TODO: Handle bake_into_final option

    # --- 8. Prepare Metadata ---
    # Construct the detailed metadata dict as per Section 10 / 16.4
    logger.info("Preparing metadata...")
    sample.metadata = {
        "generator_version": "0.1.0", # TODO: Get version dynamically
        "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": None, # Will be filled at the end
        "seed_used": sample_seed,
        "input_config_snapshot": config, # Store the input config
        "output_parameters": {
            "image_width": width, "image_height": height,
            "bit_depth": config.get('bit_depth', 16),
            "pixel_size_nm": config.get('pixel_size_nm')
        },
        "background": bg_config,
        "layers": [], # TODO: Populate with detailed layer/shape/artifact info collected above
        "applied_global_effects": applied_effects_list,
        "output_files": {} # Will be populated by save_sample
    }

    # --- 9. Save Outputs ---
    logger.info("Saving output files...")
    save_sample(sample, config, main_file_output_dir, subdir_output_path, file_suffix, logger) # Pass config to know what to save and bit depth

    # --- 10. Finalize ---
    end_time = time.time()
    runtime = round(end_time - start_time, 3)
    sample.metadata["runtime_seconds"] = runtime
    logger.info(f"Sample generation completed in {runtime:.3f}s")
    
    # --- Calculate and save hashes ---
    if output_opts.get('save_hashes', False):
         logger.info("Calculating file hashes...")
         hashes = calculate_hashes(sample.output_paths, main_file_output_dir) # Use helper
         h_fname = "hashes.json"
         h_fpath = os.path.join(subdir_output_path_for_log, h_fname)
         save_metadata(hashes, h_fpath) # Use save_metadata for json
         paths = sample.output_paths # Get paths dict
         paths['hashes'] = os.path.join(subdir_output_path, h_fname) # Add relative path
         
    # Re-save metadata with runtime and potentially hash path
    if 'metadata' in sample.output_paths and sample.output_paths['metadata']:
        meta_relative_path = sample.output_paths['metadata']
        meta_full_path = os.path.join(main_file_output_dir, meta_relative_path)
        logger.debug(f"Updating metadata at full path: '{meta_full_path}'")
        save_metadata(sample.metadata, meta_full_path)
    else:
        logger.error("Could not find metadata path in sample.output_paths to save runtime.")

    print(f"Generated sample {sample_id_numeric} data. Log: {log_file_path if save_logs_flag else 'Disabled'}")
    print(f"Generated sample {sample_id_numeric} data in {sample.metadata['runtime_seconds']:.3f}s")
    print(f"  Main files in: {os.path.abspath(main_file_output_dir)}")
    print(f"  Supporting files in: {os.path.abspath(subdir_output_path)}")
    
    # Close log handler for this sample
    if file_handler:
        logger.removeHandler(file_handler)
        file_handler.close()

    return sample


def save_sample(sample: GeneratedSample, config: Dict[str, Any], main_dir: str, sub_dir: str, file_suffix: str, logger: logging.Logger):
    """
    Saves the generated sample data to disk using the revised structure:
    - Main images/meta in main_dir (with suffix)
    - Details like masks, logs, configs in sub_dir (named sem_xxxxx)
    """
    logger.debug(f"Saving sample data. Main dir: '{main_dir}', Sub dir: '{sub_dir}'")
    # Define specific subdirectories within the sample's subdir
    layers_base_dir = os.path.join(sub_dir, "layers")
    layers_combined_dir = os.path.join(sub_dir, "layers_combined")
    logs_dir = os.path.join(sub_dir, "logs")
    ensure_dir_exists(layers_base_dir)
    ensure_dir_exists(layers_combined_dir)
    ensure_dir_exists(logs_dir) # Create only if saving logs

    # Dictionary to store relative paths (from main_dir) for metadata
    paths = {}
    sample_subdir_name = os.path.basename(sub_dir) # e.g., "sem_00001"

    # --- Get Output Options ---
    output_opts = config.get('output_options', {})
    img_format_final = output_opts.get('output_formats', {}).get('image_final', 'tif') # Format for top-level final image
    img_format_vis = output_opts.get('output_formats', {}).get('image_vis', 'png') # Format for internal visual images (clean, noisy, bg_vis)
    mask_format_data = output_opts.get('output_formats', {}).get('masks_data', 'npy') # Format for mask data (NPY recommended)
    mask_format_vis = output_opts.get('output_formats', {}).get('masks_vis', 'png') # Format for visual masks
    instance_mask_format_data = output_opts.get('output_formats', {}).get('instance_mask_data', 'tif') # TIF good for uint16
    save_per_layer_renders = output_opts.get('save_per_layer_renders', False) # Control per-layer image saving
    save_optional_npy = output_opts.get('save_optional_npy', True) # Control saving .npy files
    # Add flags for saving noise maps, warp fields etc.
    save_noise_map = output_opts.get('save_noise_map', False)
    save_warp_field = output_opts.get('save_warp_field', False)
    save_gifs = output_opts.get('save_gifs', False)
    save_hashes = output_opts.get('save_hashes', False) # Control hash saving
    save_logs = output_opts.get('save_logs', False) # Control log saving

    bit_depth_val = config.get('bit_depth', 16)
    target_dtype_final = np.uint16 if bit_depth_val == 16 else np.uint8 # For final TIF/PNG
    target_dtype_vis = np.uint8 # Visual PNGs usually uint8
    warp_field_vis_format = output_opts.get('output_formats',{}).get('warp_field_vis', 'png')

    # --- Helper function for visible masks ---
    # Helper for visual masks
    def make_mask_visible(mask_array: np.ndarray) -> Optional[np.ndarray]:
        if mask_array is None: return None
        return (mask_array > 0).astype(np.uint8) * 255

    # Helper for saving numpy arrays
    def save_npy(data_array: np.ndarray, filepath: str):
        if data_array is None: return
        ensure_dir_exists(os.path.dirname(filepath))
        try:
            np.save(filepath, data_array)
        except Exception as e:
            print(f"Error saving NPY file {filepath}: {e}")
    # --- End Helper ---

    # --- 1. Save Top-Level Final Image ---
    if sample.image_final is not None:
        fname = f"{os.path.basename(sub_dir)}.{img_format_final}" # e.g., sem_00001.tif
        fpath = os.path.join(main_dir, fname)
        logger.debug(f"Saving final image (top-level) to '{fpath}'")
        save_image(scale_to_uint(sample.image_final, target_dtype_final), fpath)
        paths['final_image_top_level'] = fname # Relative to main_dir
        
    # --- 2. Save Files in Sample Subdirectory (`sub_dir`) ---

    #   --- Root of Subdir ---
    if sample.image_clean is not None:
        fname = f"image_clean.{img_format_vis}"
        fpath = os.path.join(sub_dir, fname)
        save_image(scale_to_uint(sample.image_clean, target_dtype_vis), fpath)
        paths['image_clean'] = os.path.join(sample_subdir_name, fname)

    if sample.image_final is not None: # Save noisy visual version too
        fname = f"image_final_noisy.{img_format_vis}"
        fpath = os.path.join(sub_dir, fname)
        save_image(scale_to_uint(sample.image_final, target_dtype_vis), fpath)
        paths['image_final_noisy'] = os.path.join(sample_subdir_name, fname)

    if sample.combined_original_mask is not None:
        if save_optional_npy: save_npy(sample.combined_original_mask.astype(np.uint8), os.path.join(sub_dir, "combined_original_mask.npy"))
        save_image(make_mask_visible(sample.combined_original_mask), os.path.join(sub_dir, f"combined_original_mask_vis.{mask_format_vis}"))
        paths['combined_original_mask'] = os.path.join(sample_subdir_name, "combined_original_mask.npy") # Point to data
        paths['combined_original_mask_vis'] = os.path.join(sample_subdir_name, f"combined_original_mask_vis.{mask_format_vis}")

    if sample.combined_actual_mask is not None:
        if save_optional_npy: save_npy(sample.combined_actual_mask.astype(np.uint8), os.path.join(sub_dir, "combined_actual_mask.npy"))
        save_image(make_mask_visible(sample.combined_actual_mask), os.path.join(sub_dir, f"combined_actual_mask_vis.{mask_format_vis}"))
        paths['combined_actual_mask'] = os.path.join(sample_subdir_name, "combined_actual_mask.npy")
        paths['combined_actual_mask_vis'] = os.path.join(sample_subdir_name, f"combined_actual_mask_vis.{mask_format_vis}")

    if sample.instance_mask is not None and output_opts.get('generate_instance_masks', False):
        # Save data mask (TIF recommended for uint16)
        inst_fname_data = f"instance_mask.{instance_mask_format_data}"
        inst_fpath_data = os.path.join(sub_dir, inst_fname_data)
        save_image(sample.instance_mask, inst_fpath_data) # Save raw uint16/32 data
        paths['instance_mask'] = os.path.join(sample_subdir_name, inst_fname_data)
        # Save visual mask
        # (Colorized version generation logic needs to be here or called from here)
        max_id = np.max(sample.instance_mask)
        vis_instance_mask = np.zeros((sample.instance_mask.shape[0], sample.instance_mask.shape[1], 3), dtype=np.uint8)
        if max_id > 0:
            hue = ((sample.instance_mask * (180.0 / (max_id + 1))) % 180).astype(np.uint8)
            saturation = np.full_like(hue, 200); saturation[sample.instance_mask == 0] = 0
            value = np.full_like(hue, 200); value[sample.instance_mask == 0] = 0
            hsv_mask = cv2.merge([hue, saturation, value])
            vis_instance_mask = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
        inst_fname_vis = f"instance_mask_vis.{mask_format_vis}"
        inst_fpath_vis = os.path.join(sub_dir, inst_fname_vis)
        save_image(vis_instance_mask, inst_fpath_vis)
        paths['instance_mask_vis'] = os.path.join(sample_subdir_name, inst_fname_vis)


    if save_noise_map and sample.noise_mask is not None: # Assuming noise stored in sample.noise_mask
         if save_optional_npy: save_npy(sample.noise_mask, os.path.join(sub_dir, "noise_map_added.npy"))
         # Scale noise map [-X, +X] -> [0, 255] for visualization
         noise_vis = sample.noise_mask
         min_n, max_n = np.min(noise_vis), np.max(noise_vis)
         if max_n > min_n: noise_vis = ((noise_vis - min_n) / (max_n - min_n)) * 255
         else: noise_vis = np.zeros_like(noise_vis) + 128 # Gray if flat
         save_image(noise_vis.astype(np.uint8), os.path.join(sub_dir, f"noise_map_added_vis.{mask_format_vis}"))
         paths['noise_map_added'] = os.path.join(sample_subdir_name, "noise_map_added.npy")
         paths['noise_map_added_vis'] = os.path.join(sample_subdir_name, f"noise_map_added_vis.{mask_format_vis}")

    # --- Save Warp Field (in sub_dir) IF REQUESTED and AVAILABLE ---
    if save_warp_field and sample.warp_field is not None:
        logger.debug("Saving displacement field.")
        wf_npy_fname = "warp_field.npy"
        wf_npy_fpath = os.path.join(sub_dir, wf_npy_fname)
        if save_optional_npy:
            save_npy(sample.warp_field, wf_npy_fpath)
            paths['warp_field'] = os.path.join(sample_subdir_name, wf_npy_fname)

        # --- Generate and Save Warp Field Visualization ---
        try:
            # Calculate expected max displacement (e.g., from config if available, or auto)
            # max_disp_param = config.get('artifact_raffle',{}).get('image_level',{}).get('effects',{}).get('elastic_mesh_deform',{}).get('parameter_ranges',{}).get('amplitude',[1.0, 15.0])[-1] # Get max possible amplitude
            # For now, just auto-scale based on the field itself by passing None
            max_disp_param = None

            logger.debug("Generating warp field visualization...")
            warp_vis = visualize_warp_field(sample.warp_field, max_expected_disp=max_disp_param)

            wf_vis_fname = f"warp_field_vis.{warp_field_vis_format}"
            wf_vis_fpath = os.path.join(sub_dir, wf_vis_fname)
            save_image(warp_vis, wf_vis_fpath) # Saves the BGR image
            paths['warp_field_vis'] = os.path.join(sample_subdir_name, wf_vis_fname)
            logger.debug(f"Saved warp field visualization to {wf_vis_fpath}")

        except Exception as e:
             logger.error(f"Error generating/saving warp field visualization: {e}")
             import traceback
             logger.error(traceback.format_exc())
         
    if save_gifs:
        print("DEBUG save_sample: Generating and saving GIFs...")
        # GIF for Actual Masks Build-up
        actual_mask_frames = [make_mask_visible(m) for m in sample.cumulative_masks if m is not None]
        if actual_mask_frames:
            gif_fname = "layers_actual_masks.gif"
            gif_fpath = os.path.join(layers_combined_dir, gif_fname)
            save_gif(actual_mask_frames, gif_fpath, fps=2) # Slow FPS for layers
            paths['layers_actual_masks_gif'] = os.path.join(sample_subdir_name, "layers_combined", gif_fname)

        # GIF for Original Masks Build-up (Requires generating cumulative original masks)
        # TODO: Generate cumulative original masks in pipeline if this GIF is desired
        # original_mask_frames = [make_mask_visible(m) for m in sample.cumulative_original_masks if m is not None]
        # if original_mask_frames:
        #    gif_fname = "layers_original_masks.gif"
        #    # ... save gif ...
        #    paths['layers_original_masks_gif'] = ...
        else:
            print("DEBUG save_sample: Cumulative original masks not available for GIF.")
         
    # Save specific config for this sample
    cfg_fname = "configuration.json"
    cfg_fpath = os.path.join(sub_dir, cfg_fname)
    save_metadata(config, cfg_fpath) # Save the sample-specific config used
    paths['configuration'] = os.path.join(sample_subdir_name, cfg_fname)

    # Save seed text file
    seed_fname = "seed.txt"
    seed_fpath = os.path.join(sub_dir, seed_fname)
    save_text(str(sample.metadata['seed_used']), seed_fpath)
    paths['seed_file'] = os.path.join(sample_subdir_name, seed_fname)
    
    #   --- /layers_combined/ Subdir ---
    if sample.background is not None:
        if save_optional_npy: save_npy(sample.background, os.path.join(layers_combined_dir, "background.npy"))
        save_image(scale_to_uint(sample.background, target_dtype_vis), os.path.join(layers_combined_dir, f"background_vis.{img_format_vis}"))
        paths['background_data'] = os.path.join(sample_subdir_name, "layers_combined", "background.npy")
        paths['background_vis'] = os.path.join(sample_subdir_name, "layers_combined", f"background_vis.{img_format_vis}")

    for i, cum_mask in enumerate(sample.cumulative_masks):
         if cum_mask is not None:
            if save_optional_npy: save_npy(cum_mask.astype(np.uint8), os.path.join(layers_combined_dir, f"cumulative_actual_mask_{i:02d}.npy"))
            save_image(make_mask_visible(cum_mask), os.path.join(layers_combined_dir, f"cumulative_actual_mask_{i:02d}_vis.{mask_format_vis}"))
            paths[f'cumulative_actual_mask_{i:02d}'] = os.path.join(sample_subdir_name, "layers_combined", f"cumulative_actual_mask_{i:02d}.npy")
            paths[f'cumulative_actual_mask_{i:02d}_vis'] = os.path.join(sample_subdir_name, "layers_combined", f"cumulative_actual_mask_{i:02d}_vis.{mask_format_vis}")
    
    #   --- /layers/layer_xx/ Subdirs ---
    for i, (orig_mask, actual_mask) in enumerate(sample.layer_masks):
        layer_xx_dir = os.path.join(layers_base_dir, f"layer_{i:02d}")
        ensure_dir_exists(layer_xx_dir)
        layer_rel_path = os.path.join("layers", f"layer_{i:02d}") # Relative path for metadata

        if orig_mask is not None:
            if save_optional_npy: save_npy(orig_mask.astype(np.uint8), os.path.join(layer_xx_dir, "original_mask.npy"))
            save_image(make_mask_visible(orig_mask), os.path.join(layer_xx_dir, f"original_mask_vis.{mask_format_vis}"))
            paths[f'layer_{i:02d}_original_mask'] = os.path.join(sample_subdir_name, layer_rel_path, "original_mask.npy")
            paths[f'layer_{i:02d}_original_mask_vis'] = os.path.join(sample_subdir_name, layer_rel_path, f"original_mask_vis.{mask_format_vis}")
            # Optional original render
            # if save_per_layer_renders and corresponding data exists in sample: save here...

        if actual_mask is not None:
             if save_optional_npy: save_npy(actual_mask.astype(np.uint8), os.path.join(layer_xx_dir, "actual_mask.npy"))
             save_image(make_mask_visible(actual_mask), os.path.join(layer_xx_dir, f"actual_mask_vis.{mask_format_vis}"))
             paths[f'layer_{i:02d}_actual_mask'] = os.path.join(sample_subdir_name, layer_rel_path, "actual_mask.npy")
             paths[f'layer_{i:02d}_actual_mask_vis'] = os.path.join(sample_subdir_name, layer_rel_path, f"actual_mask_vis.{mask_format_vis}")
             # Optional actual render
             # if save_per_layer_renders and corresponding data exists in sample: save here...
             # Optional defect mask
             # if output_opts.get('save_defect_mask', False) and orig_mask is not None: calculate and save XOR mask here...

    # --- Update paths dict in sample object ---
    # This will be saved again when runtime is added to metadata
    sample.output_paths = paths
    sample.metadata['output_files'] = paths

    # --- OPTIONAL: Calculate and save hashes ---
    if save_hashes:
         print("DEBUG save_sample: Calculating file hashes...")
         # TODO: Implement hash calculation (e.g., sha256) for key files listed in paths
         # Store hashes in hashes.json within sub_dir
         # paths['hashes'] = os.path.join(sample_subdir_name, "hashes.json")
         pass # Placeholder

    # --- OPTIONAL: Save Logs ---
    if save_logs:
         print("DEBUG save_sample: Saving generation log...")
         # TODO: Implement proper logging redirection per sample
         # Save log content to logs/generation.log within sub_dir
         # paths['log_file'] = os.path.join(sample_subdir_name, "logs", "generation.log")
         pass # Placeholder

    # Update the paths dict in the sample object *before* it's saved in metadata again later
    sample.output_paths = paths
    sample.metadata['output_files'] = paths # Ensure metadata reflects final paths
    
def calculate_hashes(output_paths: Dict[str, str], base_dir: str, algo: str = 'sha256') -> Dict[str, str]:
    """Calculates hashes for files listed in output_paths."""
    hashes = {}
    for key, rel_path in output_paths.items():
        # Skip metadata/hash file itself? Or include? Including for now.
        # if key in ['metadata', 'hashes', 'configuration', 'seed_file']: continue
        full_path = os.path.join(base_dir, rel_path)
        if os.path.isfile(full_path):
            try:
                 hasher = hashlib.new(algo)
                 with open(full_path, 'rb') as f:
                      while True:
                           chunk = f.read(4096) # Read in chunks
                           if not chunk: break
                           hasher.update(chunk)
                 hashes[rel_path] = hasher.hexdigest()
            except Exception as e:
                 print(f"Warning: Could not calculate hash for {rel_path}: {e}")
                 hashes[rel_path] = f"Error: {e}"
        else:
             hashes[rel_path] = "File not found or not a file"
    return hashes