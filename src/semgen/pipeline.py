import numpy as np
import cv2
import os
import time
import random
from typing import Dict, Any, Optional, List, Tuple

from .core import GeneratedSample
from .config_loader import load_config # Assuming using this loader
from .raffle import Raffler
from .shapes import create_shape
from .artifacts import (create_artifact, AffineTransform, ElasticMeshDeform)
from .noise import create_noise
from .utils import (save_image, save_metadata, save_text, ensure_dir_exists,
                   generate_cumulative_mask, combine_masks, scale_to_uint)
from .artifacts.shape_level import EdgeRipple, BreaksHoles

# Potentially import background generation functions if they become complex
# from .background import generate_background

# Placeholder for background generation
def generate_background(config: Dict[str, Any], size: Tuple[int, int]) -> np.ndarray:
    bg_type = config.get('background_type', 'flat')
    height, width = size
    # Default to float32 [0,1] for internal processing
    background = np.zeros((height, width), dtype=np.float32)

    if bg_type == 'flat':
        intensity = float(config.get('flat_intensity', 0.1))
        background.fill(intensity)
    elif bg_type == 'gradient':
        # config gradient params should already be floats/chosen style
        params = config.get('gradient_params', {})
        start = float(params.get('start_intensity', 0.0))
        end = float(params.get('end_intensity', 0.2))
        style = config.get('gradient_style', 'linear') # Style was already chosen
        if style == 'linear':
             y_coords = np.linspace(start, end, height)
             background = np.tile(y_coords, (width, 1)).T
        else: # radial etc. - TODO
             print("Warning: Radial gradient background not fully implemented.")
             background.fill((start + end)/2)
    elif bg_type == 'noise':
        # Get parameters specific to the noise background type
        # These should have been randomized to single values by config_randomizer
        noise_amplitude = float(config.get('noise_amplitude', 0.05)) # Safe if value is float/int
        noise_type = config.get('noise_type', 'gaussian') # Get chosen noise type
        # Define a base level - use 0.0 or a dedicated 'noise_base_intensity' param?
        # Let's use a default base or potentially randomize it if added to config
        noise_base = 0.05 # Or maybe randomize config.get('noise_base_intensity', [0.0, 0.1])?

        print(f"Generating '{noise_type}' noise background (Amplitude: {noise_amplitude:.3f}, Base: {noise_base:.3f})")

        # TODO: Implement different noise types (Perlin/Simplex needed)
        if noise_type == 'gaussian':
            noise = np.random.normal(loc=0.0, scale=noise_amplitude, size=size)
            background = noise_base + noise
        elif noise_type in ['perlin', 'simplex']:
             print(f"Warning: Background noise type '{noise_type}' not implemented, using Gaussian.")
             noise = np.random.normal(loc=0.0, scale=noise_amplitude, size=size)
             background = noise_base + noise
        else: # Fallback
            print(f"Warning: Unknown background noise type '{noise_type}', using Gaussian.")
            noise = np.random.normal(loc=0.0, scale=noise_amplitude, size=size)
            background = noise_base + noise

        # Clip the final noise background
        np.clip(background, 0.0, 1.0, out=background)
    else: # composite / unknown
        print(f"Warning: Unknown background type '{bg_type}', using flat default.")
        background.fill(0.1)

    return background

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


        # Draw bar and label if bar fits
        if bar_length_px > 0 and bar_x >= 0 and (bar_x + bar_length_px) <= width and bar_y >= 0 and (bar_y + bar_thickness_px) <= height:
            cv2.rectangle(meta_overlay, (bar_x, bar_y), (bar_x + bar_length_px, bar_y + bar_thickness_px),
                          bar_color, thickness=-1) # Filled rectangle
            cv2.putText(meta_overlay, scale_label, (label_x, label_y), font_face, font_scale,
                        font_color, font_thickness, lineType=cv2.LINE_AA)
        else:
             print(f"Warning: Scale bar ({bar_length_px}px) does not fit or pixel_size_nm invalid.")
             bar_drawn_height = 0 # Reset height if not drawn

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

    # --- 1. Initialization ---
    sample_id_numeric = config.get('start_index', 0) + sample_index
    sample_id_str = f"sem_{sample_id_numeric:05d}" # ID for subdirectory
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
    raffler = Raffler(config.get('artifact_raffle', {}), rng)
    applied_effects_list = raffler.raffle_effects_for_image() # List of {'type':..., 'parameters':..., 'order':...}

    # --- 3. Background Generation ---
    bg_config = config.get('background', {})
    sample.background = generate_background(bg_config, size)
    # Start with the background for clean image
    # Work with float32 internally
    current_image = sample.background.astype(np.float32).copy()

    # --- 4. Layer & Shape Generation ---
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
        print(f"-- Processing Layer {layer_idx}, Type: {layer_conf.get('shape_type')} --")
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

        elif pattern_type == 'random_scatter':
            num_shapes = int(layer_conf.get('num_shapes', 10))
            for _ in range(num_shapes):
                 cx = rng.uniform(0, width)
                 cy = rng.uniform(0, height)
                 shape_instance_configs.append({"center": (cx, cy)})

        # TODO: Implement hex_grid, radial_grid
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
                 # print(f"DEBUG: Layer {layer_idx}, Instance {instance_idx}, Orig Mask Sum: {np.sum(instance_original_mask)}")
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

            # print(f"DEBUG: Layer {layer_idx}, Instance {instance_idx}, Actual Mask Sum: {np.sum(instance_actual_mask)}")
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
        print("Generating instance mask...")
        instance_mask_img = np.zeros(size, dtype=np.uint16) # Use uint16 for more IDs
        current_id = 1
        # Use the per-instance actual masks we stored
        for inst_mask in all_shapes_final_actual_masks:
            if inst_mask is not None and np.any(inst_mask):
                 instance_mask_img[inst_mask > 0] = current_id
                 current_id += 1
        sample.instance_mask = instance_mask_img
        # print(f"DEBUG: Instance Mask Max ID: {np.max(instance_mask_img)}")
        if sample.instance_mask is not None:
             # Instance mask saving logic will be handled inside save_sample
             pass # No saving here, just generation
        # print(f"DEBUG: Instance Mask Max ID: {np.max(instance_mask_img)}")
        # TODO: Save instance mask in save_sample
        ensure_dir_exists(os.path.join(sample.output_dir, "masks", "instance_masks")) # Redundant? Ensure it exists
        inst_fname = f"instance_mask.{mask_format}"
        inst_fpath = os.path.join(sample.output_dir, "masks", "instance_masks", inst_fname)
        # print(f"DEBUG: Saving instance mask to '{inst_fpath}'")
        save_image(sample.instance_mask, inst_fpath) # Save as uint16
        sample.output_paths['instance_mask'] = os.path.join(os.path.basename(sample.output_dir), "masks", "instance_masks", inst_fname) # Relative path


    # TODO: Implement defect mask generation more robustly

    # --- 6. Apply Image-Level Artifacts & Noise ---
    final_image = sample.image_clean.copy()
    current_actual_masks = sample.get_actual_masks() # Keep track of masks needing geom warp

    for effect_info in applied_effects_list:
        category = effect_info['category']
        artifact_type = effect_info['type']
        parameters = effect_info['parameters'] # Get the parameters dict
        # print(f"DEBUG: Applying global effect: {artifact_type} (Category: {category})")
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

                # Apply based on type
                if isinstance(instance, (AffineTransform, ElasticMeshDeform)): # Geometric warps
                    # print(f"DEBUG: Applying geometric warp '{artifact_type}'...")
                    # Apply the warp and unpack the returned tuple
                    warped_image, warped_masks = instance.apply(final_image, masks=current_actual_masks) 
                    final_image = warped_image     # final_image remains an ndarray
                    current_actual_masks = warped_masks # Update the list of masks
                    # print(f"DEBUG: Geometric warp '{artifact_type}' applied.")
                    # Also warp combined/instance/defect masks if they exist
                    if sample.combined_actual_mask is not None:
                         # print("DEBUG: Warping combined_actual_mask...")
                         # Apply to the mask, ignore the returned image (pass mask as image)
                         # Pass the mask in a list, unpack the list
                         _, [warped_combined_actual] = instance.apply(sample.combined_actual_mask, masks=[sample.combined_actual_mask])
                         sample.combined_actual_mask = warped_combined_actual
                    if sample.instance_mask is not None and output_opts.get('generate_instance_masks', False):
                         # print("DEBUG: Warping instance_mask...")
                          # Instance masks need nearest neighbor interpolation during warp
                          # We might need to pass interpolation hints to apply method?
                          # For now, assume apply handles masks correctly or modify apply.
                          # Let's assume apply uses nearest for masks passed in the list.
                         _, [warped_instance_mask] = instance.apply(sample.instance_mask, masks=[sample.instance_mask])
                         sample.instance_mask = warped_instance_mask
                    # TODO: Warp defect mask similarly if it exists
                else: # Intensity, blur, noise artifacts
                    # print(f"DEBUG: Applying intensity/noise effect '{artifact_type}'...")
                    # These return only the image, assignment is correct
                    # Ensure the instance.apply method ONLY returns the image ndarray
                    result = instance.apply(final_image)
                    if isinstance(result, tuple):
                         print(f"ERROR: Intensity/Noise artifact '{artifact_type}' unexpectedly returned a tuple!")
                         # Handle error appropriately, maybe take first element?
                         final_image = result[0] # Risky assumption
                    else:
                         final_image = result
                    # print(f"DEBUG: Effect '{artifact_type}' applied.")

            except Exception as e:
                 print(f"Error applying global effect {artifact_type}: {e}")
                 import traceback
                 traceback.print_exc() # Print full traceback for errors here

    # Now final_image should always be an ndarray when assigned below
    sample.image_final = final_image

    # --- 7. Generate Overlays ---
    if output_opts.get('save_debug_overlays', True): # Check default if needed
        sample.overlay_image = generate_debug_overlay(sample.image_final, current_actual_masks) # Use warped masks
    meta_overlay_conf = config.get('metadata_overlay', {})
    sample.metadata_overlay_image = generate_metadata_overlay(meta_overlay_conf, config, size)
        # TODO: Handle bake_into_final option

    # --- 8. Prepare Metadata ---
    # Construct the detailed metadata dict as per Section 10 / 16.4
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
    save_sample(sample, config, main_file_output_dir, subdir_output_path, file_suffix) # Pass config to know what to save and bit depth

    # --- 10. Finalize ---
    end_time = time.time()
    sample.metadata["runtime_seconds"] = round(end_time - start_time, 3)
    # print(f"DEBUG pipeline: Updating metadata at path: '{sample.output_paths.get('metadata', 'METADATA PATH NOT FOUND')}'") # ADD DEBUG
    if 'metadata' in sample.output_paths and sample.output_paths['metadata']:
        meta_full_path = os.path.join(subdir_output_path, sample.output_paths['metadata'])
        # print(f"DEBUG pipeline: Reconstructed metadata full path: '{meta_full_path}'") # ADD DEBUG
        save_metadata(sample.metadata, meta_full_path)
    else:
        print("ERROR: Could not find metadata path in sample.output_paths to save runtime.")

    print(f"Generated sample {sample_id_numeric} data in {sample.metadata['runtime_seconds']:.3f}s")
    print(f"  Main files in: {os.path.abspath(main_file_output_dir)}")
    print(f"  Supporting files in: {os.path.abspath(subdir_output_path)}")

    return sample


def save_sample(sample: GeneratedSample, config: Dict[str, Any], main_dir: str, sub_dir: str, file_suffix: str):
    """Saves the generated sample data to disk."""
    masks_dir = os.path.join(sub_dir, "masks")
    paths = {} # Store actual *relative* paths from main_dir

    # Get output options from the main config dict
    output_opts = config.get('output_options', {})
    img_format = output_opts.get('output_formats', {}).get('image_final', 'tif')
    mask_format = output_opts.get('output_formats', {}).get('masks', 'tif')
    bit_depth_val = config.get('bit_depth', 16)
    target_dtype = np.uint16 if bit_depth_val == 16 else np.uint8
    mask_save_dtype = np.uint8

    # Save final image
    if sample.image_final is not None:
        fname = f"image_final{file_suffix}.{img_format}"
        fpath = os.path.join(main_dir, fname)
        # print(f"DEBUG save_sample: Saving image_final to '{fpath}'")
        save_image(scale_to_uint(sample.image_final, target_dtype), fpath)
        paths['image_final'] = fname # Relative to main_dir

    fname_meta = f"meta{file_suffix}.json"
    fpath_meta = os.path.join(sub_dir, fname_meta)
    # print(f"DEBUG save_sample: Saving metadata to '{fpath_meta}'")
    sample.metadata['output_files'] = paths # Record paths saved *so far*
    save_metadata(sample.metadata, fpath_meta)
    paths['metadata'] = fname_meta # Store final relative path

    if sample.overlay_image is not None and output_opts.get('save_debug_overlays', False):
         fname = f"overlay{file_suffix}.png"
         fpath = os.path.join(sub_dir, fname)
         # print(f"DEBUG save_sample: Saving overlay to '{fpath}'")
         save_image(sample.overlay_image, fpath)
         paths['overlay'] = fname

    if sample.metadata_overlay_image is not None and config.get('metadata_overlay',{}).get('enabled', False):
         fname = f"metadata_overlay{file_suffix}.png"
         fpath = os.path.join(sub_dir, fname)
         # print(f"DEBUG save_sample: Saving metadata overlay to '{fpath}'")
         save_image(sample.metadata_overlay_image, fpath)
         paths['metadata_overlay'] = fname

    # --- Save Supporting Files (in sub_dir) ---
    sample_subdir_name = os.path.basename(sub_dir) # e.g., "sem_00000"

    if output_opts.get('save_intermediate_images', False):
        if sample.image_clean is not None:
             fname = f"image_clean.{img_format}"
             fpath = os.path.join(sub_dir, fname)
             # print(f"DEBUG save_sample: Saving image_clean to '{fpath}'")
             save_image(scale_to_uint(sample.image_clean, target_dtype), fpath)
             paths['image_clean'] = os.path.join(sample_subdir_name, fname) # Path relative to main_dir
        if sample.background is not None:
             fname = f"background.{img_format}"
             fpath = os.path.join(sub_dir, fname)
             # print(f"DEBUG save_sample: Saving background to '{fpath}'")
             save_image(scale_to_uint(sample.background, target_dtype), fpath)
             paths['background'] = os.path.join(sample_subdir_name, fname)
             
    # --- Add saving logic for instance mask ---
    if output_opts.get('generate_instance_masks', False) and sample.instance_mask is not None:
        inst_masks_dir = os.path.join(masks_dir, "instance_masks") # Save inside masks dir
        ensure_dir_exists(inst_masks_dir) # Ensure it exists
        inst_fname = f"instance_mask.{mask_format}" # Usually tif for uint16
        inst_fpath = os.path.join(inst_masks_dir, inst_fname)
        # print(f"DEBUG: Saving instance mask to '{inst_fpath}'")
        save_image(sample.instance_mask, inst_fpath) # Save as uint16
        # Store relative path from main_dir
        sample_subdir_name = os.path.basename(sub_dir)
        paths['instance_mask'] = os.path.join(sample_subdir_name, "masks", "instance_masks", inst_fname) # Relative path

    # Save masks (in sub_dir/masks)
    for i, (orig_mask, actual_mask) in enumerate(sample.layer_masks):
        if orig_mask is not None:
            fname = f"layer_{i:02d}_original.{mask_format}"
            fpath = os.path.join(masks_dir, fname)
            # print(f"DEBUG save_sample: Saving layer {i} original mask to '{fpath}'")
            save_image(orig_mask.astype(mask_save_dtype), fpath)
            paths[f'mask_layer_{i:02d}_original'] = os.path.join(sample_subdir_name, "masks", fname)
        if actual_mask is not None:
            fname = f"layer_{i:02d}_actual.{mask_format}"
            fpath = os.path.join(masks_dir, fname)
            # print(f"DEBUG save_sample: Saving layer {i} actual mask to '{fpath}'")
            save_image(actual_mask.astype(mask_save_dtype), fpath)
            paths[f'mask_layer_{i:02d}_actual'] = os.path.join(sample_subdir_name, "masks", fname)

    for i, cum_mask in enumerate(sample.cumulative_masks):
         if cum_mask is not None:
            fname = f"layer_{i:02d}_cumulative.{mask_format}"
            fpath = os.path.join(masks_dir, fname)
            # print(f"DEBUG save_sample: Saving cumulative mask {i} to '{fpath}'")
            save_image(cum_mask.astype(mask_save_dtype), fpath)
            paths[f'mask_layer_{i:02d}_cumulative'] = os.path.join(sample_subdir_name, "masks", fname)

    if sample.combined_original_mask is not None:
        fname = f"combined_original.{mask_format}"
        fpath = os.path.join(masks_dir, fname)
        # print(f"DEBUG save_sample: Saving combined original mask to '{fpath}'")
        save_image(sample.combined_original_mask.astype(mask_save_dtype), fpath)
        paths['combined_original'] = os.path.join(sample_subdir_name, "masks", fname)
    if sample.combined_actual_mask is not None:
        fname = f"combined_actual.{mask_format}"
        fpath = os.path.join(masks_dir, fname)
        # print(f"DEBUG save_sample: Saving combined actual mask to '{fpath}'")
        save_image(sample.combined_actual_mask.astype(mask_save_dtype), fpath)
        paths['combined_actual'] = os.path.join(sample_subdir_name, "masks", fname)

    # TODO: Save instance mask, defect mask, noise mask similarly into sub_dir/masks or sub_dir

    # Save seed and elapsed time text files in sub_dir
    fname_seed = "seed.txt"
    fpath_seed = os.path.join(sub_dir, fname_seed)
    # print(f"DEBUG save_sample: Saving seed text to '{fpath_seed}'")
    save_text(str(sample.metadata['seed_used']), fpath_seed)
    paths['seed_file'] = os.path.join(sample_subdir_name, fname_seed)

    # Update the paths dict in the sample object *before* it's saved in metadata again later
    sample.output_paths = paths
    sample.metadata['output_files'] = paths # Ensure metadata reflects final paths