import random
import numpy as np
from typing import Dict, Any, List, Tuple, Union, Callable

class Raffler:
    """
    Handles the random selection and parameterization of artifacts and noise
    for a single synthetic image generation based on user configuration.
    """

    # Define a suggested logical order for applying effects
    LOGICAL_EFFECT_ORDER = [
        # --- Shape Level ---
        "elastic_deformation_local", # Affects geometry first
        "segment_displacement",
        "edge_ripple",
        "contour_smoothing",
        "breaks_holes",             # Structural changes
        "local_brightness_thickness", # Intensity change on shape
        # --- Layer Composition Happens Here ---
        # --- Image Level Geometric ---
        "affine",
        "perspective",              # If implemented
        "elastic_mesh_deform",
        "rolling_banding",
        "striping_smearing_geom",   # Geometric part of striping/smearing
        # --- Image Level Intensity/Blur/PSF ---
        "probe_psf",                # Blurring often comes early in intensity changes
        "defocus_blur",
        "topographic_shading",
        "gradient_illumination",
        "charging",                 # Often prominent, applied before general noise
        "striping_smearing_intensity", # Intensity part
        "detector_fixed_pattern",   # Static detector noise
        # --- General Noise ---
        "poisson",                  # Signal dependent first
        "sem_texture",
        "custom_texture",
        "gaussian",
        "salt_pepper",              # Impulse noise often last before quantization
        "blur",                     # General blur artifact
        "scanline_drop",            # Simulates read-out issue
        # --- Final Stage ---
        "quantization"
    ]

    def __init__(self, config: Dict[str, Any], rng: random.Random):
        """
        Initialize the Raffler.

        Args:
            config (Dict[str, Any]): The 'artifact_raffle' section of the main config.
            rng (random.Random): A seeded random number generator instance for reproducibility.
        """
        self.config = config
        self.rng = rng
        self.application_order_mode = config.get('artifact_application_order', 'logical').lower()

    def _randomize_parameter(self, param_config: Union[list, tuple, int, float, str, bool, dict]) -> Any:
        """
        Generates a concrete value for a parameter based on its configuration.

        Args:
            param_config: The configuration for the parameter, which can be:
                          - A list/tuple [min, max] for numeric ranges.
                          - A list of strings for categorical choices.
                          - A single value (int, float, str, bool) for fixed parameters.
                          - A dictionary (potentially for nested complex params - less common here).

        Returns:
            A single randomized value for the parameter.
        """
        if isinstance(param_config, (list, tuple)):
            if not param_config:
                 raise ValueError("Parameter range list/tuple cannot be empty.")

            first_item = param_config[0]
            if isinstance(first_item, str):
                # Categorical choice from a list of strings
                return self.rng.choice(param_config)
            elif isinstance(first_item, (int, float)):
                # Numeric range [min, max]
                if len(param_config) != 2:
                    raise ValueError(f"Numeric range must have 2 elements [min, max], got: {param_config}")
                min_val, max_val = param_config
                if isinstance(first_item, int) and isinstance(param_config[1], int):
                    return self.rng.randint(min_val, max_val)
                else:
                    return self.rng.uniform(float(min_val), float(max_val))
            else:
                 raise ValueError(f"Unsupported type in parameter range list: {type(first_item)}")

        elif isinstance(param_config, (int, float, str, bool)):
            # Fixed value
            return param_config
        elif isinstance(param_config, dict):
             # Could handle nested randomization if needed, but typically ranges are lists/fixed values
             print(f"Warning: Dictionary parameter config not fully supported for randomization: {param_config}. Returning as is.")
             return param_config
        else:
             raise ValueError(f"Unsupported parameter configuration type: {type(param_config)}")


    def raffle_effects_for_image(self) -> List[Dict[str, Any]]:
        """
        Performs the raffle for one image, selecting effects and their parameters.

        Returns:
            List[Dict[str, Any]]: A list of selected effect dictionaries, each containing
                                  'type' (str, effect name) and 'parameters' (dict).
                                  The list is ordered according to the application order mode.
        """
        selected_effects_by_category = {}
        all_selected_effects = []

        # Categories defined in the config structure (e.g., shape_level, image_level, noise, instrument_effects)
        categories = [cat for cat in self.config if isinstance(self.config[cat], dict) and 'effects' in self.config[cat]]

        for category in categories:
            category_config = self.config[category]
            max_effects = category_config.get('max_effects_per_image', float('inf'))
            effects_config = category_config.get('effects', {})
            candidates = []

            # Shuffle the order of checking effects within a category for fairness
            effect_items = list(effects_config.items())
            self.rng.shuffle(effect_items)

            for effect_name, effect_details in effect_items:
                if not effect_details.get('enabled', True):
                    continue

                probability = effect_details.get('probability', 1.0)

                # Roll the dice
                if self.rng.random() < probability:
                    # Parameterize
                    parameterized_values = {}
                    param_ranges = effect_details.get('parameter_ranges', {})
                    for param_name, param_config in param_ranges.items():
                        try:
                           parameterized_values[param_name] = self._randomize_parameter(param_config)
                        except ValueError as e:
                           print(f"Warning: Error randomizing parameter '{param_name}' for effect '{effect_name}': {e}. Skipping parameter.")


                    # Add specific type/category info if needed later
                    candidates.append({
                        "type": effect_name,
                        "category": category, # Store which category it came from
                        "parameters": parameterized_values
                    })

            # Limit the number of effects per category if needed
            if len(candidates) > max_effects:
                selected_category_effects = self.rng.sample(candidates, max_effects)
            else:
                selected_category_effects = candidates

            selected_effects_by_category[category] = selected_category_effects
            all_selected_effects.extend(selected_category_effects)

        # --- Determine Application Order ---
        if not all_selected_effects:
            return [] # No effects selected

        if self.application_order_mode == 'random':
            self.rng.shuffle(all_selected_effects)
            # Add order index for metadata
            for i, effect in enumerate(all_selected_effects):
                 effect['order'] = i
            return all_selected_effects
        else: # 'logical' order (default)
            # Create a mapping from effect type to its logical order index
            order_map = {effect_type: i for i, effect_type in enumerate(self.LOGICAL_EFFECT_ORDER)}
            unknown_order_offset = len(self.LOGICAL_EFFECT_ORDER)

            # Sort selected effects based on the defined logical order
            def sort_key(effect):
                return order_map.get(effect['type'], unknown_order_offset + self.rng.random()) # Place unknowns at end, randomized

            all_selected_effects.sort(key=sort_key)
            # Add order index for metadata
            for i, effect in enumerate(all_selected_effects):
                 effect['order'] = i
            return all_selected_effects