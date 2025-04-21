import argparse
import os
import json
import time
import random # For seeding if not provided

from ..utils import ensure_dir_exists
from ..config_loader import load_config, DEFAULT_CONFIG
from ..config_randomizer import randomize_config_for_sample # Import the new function
from ..pipeline import generate_sample
from .. import __version__

def create_parser() -> argparse.ArgumentParser:
    """Creates the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description=f"Synthetic SEM Image Generator v{__version__}. Generate realistic SEM images with annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows defaults in help
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to the JSON configuration file. If not provided, default settings are used.",
        default=None
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        help="Directory to save the generated samples. Overrides 'output_dir' in the config file.",
        default=None # Default taken from config file or DEFAULT_CONFIG
    )

    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        help="Number of samples to generate. Overrides 'num_samples' in the config file.",
        default=None # Default taken from config file or DEFAULT_CONFIG
    )

    parser.add_argument(
        "-s", "--seed",
        type=int,
        help="Master random seed for reproducibility. Overrides 'seed' in the config file. Set to 0 for random.",
        default=None # Default handling inside load_config
    )

    parser.add_argument(
        "--start-index",
        type=int,
        help="Starting index for sample numbering. Overrides 'start_index' in the config file.",
        default=None # Default taken from config file or DEFAULT_CONFIG
    )

    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show the final configuration that will be used (after loading and overrides) and exit."
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    return parser

def run_cli():
    """Parses arguments and runs the generation pipeline."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        # --- Load Base Configuration ---
        # Use load_config to get the base structure and ranges
        base_config = load_config(args.config) # Loads the json with ranges/lists
        
        # --- Apply CLI Overrides to Base Config (for overall run) ---
        if args.output_dir is not None:
            base_config['output_dir'] = args.output_dir
        if args.num_samples is not None:
            base_config['num_samples'] = args.num_samples
        if args.seed is not None:
            if args.seed == 0:
                 base_config['seed'] = random.randint(1, 2**32 - 1)
                 print(f"CLI generated random seed: {base_config['seed']}")
            else:
                base_config['seed'] = args.seed
        if args.start_index is not None:
             base_config['start_index'] = args.start_index

        # Ensure essential base config values exist
        base_config.setdefault('output_dir', "./output_random") # Default random output
        base_config.setdefault('num_samples', 10) # Default number of random samples
        base_config.setdefault('start_index', 0)
        if 'seed' not in base_config or base_config['seed'] is None:
             base_config['seed'] = random.randint(1, 2**32 - 1)

        # Show base config if requested (different from showing sample config)
        if args.show_config:
            print("--- Base Configuration (with ranges/probabilities) ---")
            print(json.dumps(base_config, indent=4, default=str))
            print("-----------------------------------------------------")
            return

        # --- Prepare for Generation ---
        num_samples_to_gen = base_config['num_samples']
        master_seed = base_config['seed']
        base_output_dir = base_config['output_dir']
        start_idx = base_config['start_index']


        print(f"Starting generation of {num_samples_to_gen} RANDOMIZED samples.")
        print(f"Using Base Config: {args.config or 'Defaults'}")
        print(f"Master Seed: {master_seed}")
        print(f"Output Directory: {os.path.abspath(base_output_dir)}")
        print("-" * 30)

        # --- Run Generation Loop with Randomization ---
        total_start_time = time.time()
        # Use a dedicated RNG for config randomization, seeded consistently
        config_rng = random.Random(master_seed)

        for i in range(num_samples_to_gen):
            sample_index_actual = start_idx + i
            print(f"Generating sample {i+1}/{num_samples_to_gen} (Index: {sample_index_actual})...")

            # 1. Randomize config for THIS sample
            print("Randomizing configuration...")
            try:
                sample_config = randomize_config_for_sample(base_config, config_rng)
                sample_config['seed'] = master_seed + sample_index_actual

                # --- ADD THIS DEBUG BLOCK ---
                print("-" * 15)
                print(f"DEBUG: Randomized Config for Sample {sample_index_actual}:")
                try:
                    pass#print(json.dumps(sample_config, indent=2, default=str)) # Print the config that will be used
                except Exception as json_e:
                    print(f"Could not serialize sample_config for printing: {json_e}")
                print("-" * 15)
                # --- END DEBUG BLOCK ---

            except ValueError as e: # Catch randomization errors specifically
                 print(f"FATAL ERROR during configuration randomization for sample {sample_index_actual}: {e}")
                 # Decide whether to stop or skip the sample
                 print("Skipping this sample due to randomization error.")
                 print("-" * 10)
                 continue # Skip to next sample
            except Exception as e: # Catch other unexpected randomization errors
                 print(f"FATAL UNEXPECTED ERROR during configuration randomization for sample {sample_index_actual}: {e}")
                 print("Skipping this sample.")
                 print("-" * 10)
                 continue # Skip to next sample


            # 2. Generate the sample using the randomized config
            try:
                 _ = generate_sample(sample_config, sample_index_actual, base_output_dir, master_seed)
            except Exception as gen_e:
                 # Print error specific to generate_sample
                 print(f"FATAL ERROR during generate_sample for sample {sample_index_actual}: {gen_e}")
                 print("Attempting to continue with the next sample...")
                 # Optionally add traceback here for detailed debugging
                 import traceback
                 traceback.print_exc()

            print("-" * 10)

        total_end_time = time.time()
        print("=" * 30)
        print(f"Generation finished.")
        print(f"Total time: {total_end_time - total_start_time:.2f} seconds.")
        print(f"Samples saved to: {os.path.abspath(base_output_dir)}")
        print("=" * 30)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except json.JSONDecodeError as e:
        print(f"Error reading config file: {e}")
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        # Catch-all for unexpected errors during generation
        print(f"An unexpected error occurred: {e}")
        # Consider adding traceback printing for debugging
        # import traceback
        # traceback.print_exc()