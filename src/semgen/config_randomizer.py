import random
import copy
from typing import Dict, Any, List, Union

def randomize_config_for_sample(base_config: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """
    Takes a base config with ranges/choices and returns a concrete config
    for a single sample generation run.

    Args:
        base_config (Dict[str, Any]): The base configuration with ranges/lists.
        rng (random.Random): A seeded random number generator instance.

    Returns:
        Dict[str, Any]: A configuration dictionary with specific values chosen.
    """
    # Deep copy to avoid modifying the original base config
    sample_config = copy.deepcopy(base_config)

    # --- Define a helper for recursive randomization ---
    def randomize_value(value: Any, parent_key: str = "<root>") -> Any: # Add key for better errors
        if isinstance(value, list):
            if not value: return value # Empty list
            first = value[0]
            # Check if it's a list for categorical choice or a numeric range
            if isinstance(first, str): # Categorical choice
                return rng.choice(value)
                
            elif isinstance(first, (list, tuple)):
                # Assume this outer list is a list of choices, and each choice is a list/tuple
                # print(f"DEBUG: Choosing one list from list of lists for key '{parent_key}'")
                return rng.choice(value)
            elif isinstance(first, (int, float)) and len(value) == 2: # Numeric range [min, max]
                min_val, max_val = value
                # Ensure second value is also numeric
                if not isinstance(max_val, (int, float)):
                     raise ValueError(f"Invalid numeric range for key '{parent_key}'. Second element is not number: {value}")
                # Check if types are compatible for range function
                if isinstance(first, int) and isinstance(max_val, int):
                    if min_val > max_val: # Ensure min <= max for randint
                         print(f"Warning: Invalid integer range for key '{parent_key}': min ({min_val}) > max ({max_val}). Using min.")
                         return min_val
                    return rng.randint(min_val, max_val)
                else:
                    # Convert both to float for uniform
                    float_min, float_max = float(min_val), float(max_val)
                    if float_min > float_max: # Ensure min <= max for uniform
                         print(f"Warning: Invalid float range for key '{parent_key}': min ({float_min}) > max ({float_max}). Using min.")
                         return float_min
                    return rng.uniform(float_min, float_max)
            # Case 3: Single numeric element list [x] - return x
            elif isinstance(first, (int, float)) and len(value) == 1:
                print(f"Warning: Parameter list for key '{parent_key}' has only one element: {value}. Using this element directly.")
                return first # Return the single element

            # Case 4: List of complex objects (like layer defs) - process outside recursion
            elif isinstance(first, dict):
                 # Cannot randomize list of dicts here, should be handled by caller
                 return value

            # Case 5: Invalid list format for randomization
            else:
                # Add specific check for non-string/list/dict/number first element
                if not isinstance(first, (str, list, tuple, dict, int, float)):
                     raise ValueError(f"First element in list for key '{parent_key}' has unsupported type '{type(first)}': {value}")
                # If it passed the initial checks but didn't match specific cases (e.g. numeric list != 2 elements)
                raise ValueError(f"Unsupported list format or content for randomization for key '{parent_key}': {value}. Check [min, max] or list of choices format.")

        elif isinstance(value, dict):
            # Recursively randomize values within the dictionary
            # Pass key name down for better error messages
            return {k: randomize_value(v, parent_key=f"{parent_key}.{k}") for k, v in value.items()}
        else:
            # Primitive value (int, float, str, bool, None) - return as is
            return value

    # --- Randomize Top-Level Simple Values ---
    # Use the helper, but some require special handling (like background type)
    try:
        # --- Randomize Top-Level & Background (as before) ---
        if isinstance(sample_config.get('pixel_size_nm'), list):
             sample_config['pixel_size_nm'] = randomize_value(sample_config['pixel_size_nm'], parent_key='pixel_size_nm')

        # --- Randomize Background ---
        bg_config = sample_config.get('background', {}) # Use get for safety
        if isinstance(bg_config.get('background_type'), list):
            chosen_bg_type = rng.choice(bg_config['background_type'])
            bg_config['background_type'] = chosen_bg_type

            if chosen_bg_type == 'flat':
                bg_config['flat_intensity'] = randomize_value(bg_config.get('flat_intensity'), parent_key='background.flat_intensity')
            elif chosen_bg_type == 'gradient':
                bg_config['gradient_style'] = randomize_value(bg_config.get('gradient_style'), parent_key='background.gradient_style')
                if isinstance(bg_config.get('gradient_params'), dict):
                    bg_config['gradient_params'] = randomize_value(bg_config['gradient_params'], parent_key='background.gradient_params')
            elif chosen_bg_type == 'noise':
                bg_config['noise_type'] = randomize_value(bg_config.get('noise_type'), parent_key='background.noise_type')
                bg_config['noise_amplitude'] = randomize_value(bg_config.get('noise_amplitude'), parent_key='background.noise_amplitude')
                bg_config['noise_frequency'] = randomize_value(bg_config.get('noise_frequency'), parent_key='background.noise_frequency')
                bg_config['noise_base_intensity'] = randomize_value(bg_config.get('noise_base_intensity'), parent_key='background.noise_base_intensity')
                bg_config['noise_octaves'] = randomize_value(bg_config.get('noise_octaves'), parent_key='background.noise_octaves')
                bg_config['noise_persistence'] = randomize_value(bg_config.get('noise_persistence'), parent_key='background.noise_persistence')
                bg_config['noise_lacunarity'] = randomize_value(bg_config.get('noise_lacunarity'), parent_key='background.noise_lacunarity')

            elif chosen_bg_type == 'composite':
                bg_config['flat_intensity'] = randomize_value(bg_config.get('flat_intensity'), parent_key='background.flat_intensity')
                bg_config['gradient_style'] = randomize_value(bg_config.get('gradient_style'), parent_key='background.gradient_style')
                bg_config['gradient_params'] = randomize_value(bg_config['gradient_params'], parent_key='background.gradient_params')
                bg_config['noise_type'] = randomize_value(bg_config.get('noise_type'), parent_key='background.noise_type')
                bg_config['noise_amplitude'] = randomize_value(bg_config.get('noise_amplitude'), parent_key='background.noise_amplitude')
                bg_config['noise_frequency'] = randomize_value(bg_config.get('noise_frequency'), parent_key='background.noise_frequency')
                bg_config['noise_base_intensity'] = randomize_value(bg_config.get('noise_base_intensity'), parent_key='background.noise_base_intensity')
                bg_config['noise_octaves'] = randomize_value(bg_config.get('noise_octaves'), parent_key='background.noise_octaves')
                bg_config['noise_persistence'] = randomize_value(bg_config.get('noise_persistence'), parent_key='background.noise_persistence')
                bg_config['noise_lacunarity'] = randomize_value(bg_config.get('noise_lacunarity'), parent_key='background.noise_lacunarity')
                # Randomize composite-specific controls
                bg_config['composite_comp1_type'] = randomize_value(bg_config.get('composite_comp1_type'), parent_key='background.composite_comp1_type')
                bg_config['composite_comp2_type'] = randomize_value(bg_config.get('composite_comp2_type'), parent_key='background.composite_comp2_type')
                bg_config['composite_combine_mode'] = randomize_value(bg_config.get('composite_combine_mode'), parent_key='background.composite_combine_mode')
                bg_config['composite_base_intensity'] = randomize_value(bg_config.get('composite_base_intensity'), parent_key='background.composite_base_intensity')
                # Randomize nested params if they exist (e.g., composite_comp1_params)
                if isinstance(bg_config.get('composite_comp1_params'), dict):
                     bg_config['composite_comp1_params'] = randomize_value(bg_config['composite_comp1_params'], parent_key='background.composite_comp1_params')
                if isinstance(bg_config.get('composite_comp2_params'), dict):
                     bg_config['composite_comp2_params'] = randomize_value(bg_config['composite_comp2_params'], parent_key='background.composite_comp2_params')
                # Also ensure the underlying gradient/noise params (if not nested) are randomized
                # This might require randomizing them always, or checking comp types
                # Safer approach: Randomize all potential sub-params always
                bg_config['gradient_style'] = randomize_value(bg_config.get('gradient_style'), parent_key='background.gradient_style')
                if isinstance(bg_config.get('gradient_params'), dict): bg_config['gradient_params'] = randomize_value(bg_config['gradient_params'], parent_key='background.gradient_params')
                bg_config['noise_type'] = randomize_value(bg_config.get('noise_type'), parent_key='background.noise_type')
                bg_config['noise_amplitude'] = randomize_value(bg_config.get('noise_amplitude'), parent_key='background.noise_amplitude')

        # --- Randomize Layering ---
        layering_config = sample_config.get('layering', {})
        base_layer_pool = [ldef for ldef in layering_config.get('layers', []) if ldef.get('enabled', True)] # Get enabled definitions from base config

        if not base_layer_pool:
             print("Warning: No enabled layer definitions found in base config layering.layers pool.")
             layering_config['num_layers'] = 0
             layering_config['layers'] = [] # Ensure layers list is empty
        else:
            # 1. Determine number of layers for this sample
            if isinstance(layering_config.get('num_layers'), list):
                num_layers_to_use = randomize_value(layering_config['num_layers'], parent_key='layering.num_layers')
            else: # Use fixed number if not a list
                num_layers_to_use = layering_config.get('num_layers', 1)

            # Ensure num_layers isn't nonsensical
            if not isinstance(num_layers_to_use, int) or num_layers_to_use < 0:
                 print(f"Warning: Invalid num_layers value: {num_layers_to_use}. Setting to 1.")
                 num_layers_to_use = 1
            layering_config['num_layers'] = num_layers_to_use # Store the chosen number

            # 2. Randomly select layer definitions from the pool
            # Use random.choices to allow picking the same definition multiple times
            chosen_base_layer_defs = rng.choices(base_layer_pool, k=num_layers_to_use)

            # 3. Randomize parameters *within* the chosen definitions
            final_sample_layer_defs = []
            for i, base_layer_def in enumerate(chosen_base_layer_defs):
                # Deep copy to avoid modifying the pool or other selected layers
                sample_layer_def = copy.deepcopy(base_layer_def)
                layer_key_prefix = f"layering.layers[{i}]" # For error reporting

                # Randomize pattern params
                if isinstance(sample_layer_def.get('pattern'), dict):
                    sample_layer_def['pattern'] = randomize_value(sample_layer_def['pattern'], parent_key=f"{layer_key_prefix}.pattern")
                # Randomize shape params
                if isinstance(sample_layer_def.get('shape_params'), dict):
                    sample_layer_def['shape_params'] = randomize_value(sample_layer_def['shape_params'], parent_key=f"{layer_key_prefix}.shape_params")
                    shape_type = sample_layer_def.get('shape_type')
                    shape_params = sample_layer_def.get('shape_params', {})

                     # Special handling for line center AFTER randomizing % values
                    if shape_type == 'line':
                        shape_params = sample_layer_def['shape_params']
                        if 'center_x%' in shape_params and 'center_y%' in shape_params:
                             img_w = sample_config.get('image_width', 512) # Use potentially randomized size if needed
                             img_h = sample_config.get('image_height', 512)
                             center_x_rel = shape_params['center_x%'] # Should be a float now
                             center_y_rel = shape_params['center_y%'] # Should be a float now
                             center_x_abs = center_x_rel * img_w
                             center_y_abs = center_y_rel * img_h
                             shape_params['center'] = (center_x_abs, center_y_abs)
                             # Optionally remove % keys
                             # del shape_params['center_x%']
                             # del shape_params['center_y%']
                        elif 'center' not in shape_params:
                             # Default if neither % nor absolute center was defined/randomized
                             img_w = sample_config.get('image_width', 512)
                             img_h = sample_config.get('image_height', 512)
                             shape_params['center'] = (img_w / 2, img_h / 2)
                    elif shape_type == 'worm':
                        if 'start_point_x%' in shape_params and 'start_point_y%' in shape_params:
                            img_w = sample_config.get('image_width', 512)
                            img_h = sample_config.get('image_height', 512)
                            # Ensure % values are randomized first if they were ranges
                            start_x_rel = shape_params['start_point_x%']
                            start_y_rel = shape_params['start_point_y%']
                            if isinstance(start_x_rel, list): start_x_rel = randomize_value(start_x_rel, f'{layer_key_prefix}.shape_params.start_point_x%')
                            if isinstance(start_y_rel, list): start_y_rel = randomize_value(start_y_rel, f'{layer_key_prefix}.shape_params.start_point_y%')

                            start_x_abs = float(start_x_rel) * img_w
                            start_y_abs = float(start_y_rel) * img_h
                            shape_params['start_point'] = (start_x_abs, start_y_abs)
                            # Optionally remove % keys
                            # del shape_params['start_point_x%']
                            # del shape_params['start_point_y%']
                        elif 'start_point' not in shape_params:
                             print(f"Warning: Worm layer {i} needs 'start_point' or 'start_point_x%'/'start_point_y%' in shape_params.")
                             # Default start point or raise error? Defaulting near center.
                             img_w = sample_config.get('image_width', 512)
                             img_h = sample_config.get('image_height', 512)
                             shape_params['start_point'] = (img_w * rng.uniform(0.4, 0.6), img_h * rng.uniform(0.4, 0.6))

                # Randomize num_shapes if scatter pattern
                if isinstance(sample_layer_def.get('num_shapes'), list):
                    sample_layer_def['num_shapes'] = randomize_value(sample_layer_def['num_shapes'], parent_key=f"{layer_key_prefix}.num_shapes")
                # Randomize alpha
                if isinstance(sample_layer_def.get('alpha'), list):
                    sample_layer_def['alpha'] = randomize_value(sample_layer_def['alpha'], parent_key=f"{layer_key_prefix}.alpha")
                # Randomize composition mode
                if isinstance(sample_layer_def.get('composition_mode'), list):
                    sample_layer_def['composition_mode'] = randomize_value(sample_layer_def['composition_mode'], parent_key=f"{layer_key_prefix}.composition_mode")

                final_sample_layer_defs.append(sample_layer_def)

            # Replace the base pool with the specific layers for this sample
            layering_config['layers'] = final_sample_layer_defs

            # Randomize layer order if needed AFTER selecting the layers
            if isinstance(layering_config.get('layer_order'), list):
                 layering_config['layer_order'] = randomize_value(layering_config['layer_order'], parent_key='layering.layer_order')
            if layering_config.get('layer_order') == 'random':
                 rng.shuffle(layering_config['layers']) # Shuffle the chosen layers

        # --- Randomize Artifact Raffle Order (as before) ---
        raffle_config = sample_config.get('artifact_raffle', {})
        if isinstance(raffle_config.get('artifact_application_order'), list):
             raffle_config['artifact_application_order'] = randomize_value(raffle_config['artifact_application_order'], parent_key='artifact_raffle.artifact_application_order')

        # --- Randomize other top-level options (as before) ---
        meta_overlay_config = sample_config.get('metadata_overlay', {})
        if isinstance(meta_overlay_config, dict): # Check if it exists and is a dict
             # Use randomize_value helper on the whole dict
             # This will recursively handle nested ranges like scale_bar.length, text_info.font_size_pt, styling offsets etc.
             sample_config['metadata_overlay'] = randomize_value(meta_overlay_config, parent_key='metadata_overlay')

    except ValueError as e:
         print(f"Configuration Randomization Error: {e}")
         raise e
    except KeyError as e:
         print(f"Configuration Randomization Error: Missing expected key {e}")
         raise e

    sample_config['num_samples'] = 1 # Config is for ONE sample run now
    return sample_config