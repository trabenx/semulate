import json
import os
import random
from typing import Dict, Any, Optional

# Placeholder for default configuration structure
# In a real application, this could be much more detailed
DEFAULT_CONFIG = {
    "image_width": 512,
    "image_height": 512,
    "bit_depth": 16,
    "num_samples": 1,
    "start_index": 0,
    "pixel_size_nm": 10.0,
    "output_dir": "./output",
    "background": {
        "background_type": "flat",
        "flat_intensity": 0.1
    },
    "layering": {
        "num_layers": 1,
        "layers": [
            {
                "shape_type": "circle",
                "pattern": {"type": "grid", "rows": 5, "cols": 5, "spacing": 64},
                "shape_params": {"radius": [15, 25], "intensity": [0.6, 0.9]},
            }
        ]
    },
    "artifact_raffle": {
        "artifact_application_order": "logical",
        "shape_level": {"max_effects_per_image": 1, "effects": {}},
        "image_level": {"max_effects_per_image": 1, "effects": {}},
        "noise": {"max_effects_per_image": 1, "effects": {}},
        "instrument_effects": {"max_effects_per_image": 1, "effects": {}}
    },
    "output_options": { # Renamed from Mask/Debug/Extra for clarity
        "save_intermediate_images": False,
        "save_noise_mask": False,
        "save_debug_overlays": True,
        "save_displacement_fields": False,
        "generate_instance_masks": False,
        "output_formats": {"image_final": "tif", "masks": "tif"},
        "use_batch_archive": False,
        "delete_on_download": False
    },
    "metadata_overlay": {
        "enabled": True,
        "scale_bar": {"length": 1.0, "units": "µm", "thickness_px": 4},
        "text_info": {"lines": ["15 kV", "x10.0k"], "font_size_pt": 10},
        "styling": {"font_color": "white", "bar_color": "white", "anchor": "bottom_right", "offset_px": [10, 10]}
    }
}

def merge_configs(default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges user config into default config."""
    merged = default.copy()
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            # User value overrides default, unless user value is None maybe?
            # For simplicity, user value always overrides.
            merged[key] = value
    return merged

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads configuration from a JSON file, merges with defaults.

    Args:
        config_path (Optional[str]): Path to the user's JSON config file.
                                      If None, returns the default config.

    Returns:
        Dict[str, Any]: The final configuration dictionary.

    Raises:
        FileNotFoundError: If config_path is provided but does not exist.
        json.JSONDecodeError: If the config file is not valid JSON.
        ValueError: For other loading issues.
    """
    final_config = DEFAULT_CONFIG.copy() # Start with defaults

    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Recursively merge user settings into defaults
            final_config = merge_configs(final_config, user_config)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error decoding JSON from {config_path}: {e.msg}", e.doc, e.pos)
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_path}: {e}")

    # TODO: Add validation using jsonschema if needed for robustness
    # try:
    #     validate(instance=final_config, schema=YOUR_SCHEMA)
    # except ValidationError as e:
    #     raise ValueError(f"Configuration validation error: {e.message}")

    # --- Resolve Seed ---
    # If seed is not provided or is None/0, generate one
    if final_config.get('seed') is None or final_config.get('seed') == 0:
         final_config['seed'] = random.randint(1, 2**32 - 1)
         print(f"Generated random seed: {final_config['seed']}")
    else:
        # Ensure seed is an integer if provided
        try:
            final_config['seed'] = int(final_config['seed'])
        except (ValueError, TypeError):
             print(f"Warning: Invalid seed value '{final_config['seed']}'. Generating random seed.")
             final_config['seed'] = random.randint(1, 2**32 - 1)


    return final_config