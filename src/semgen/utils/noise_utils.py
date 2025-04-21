# src/semgen/utils/noise_utils.py
import numpy as np
import noise

import hashlib

def generate_procedural_noise_2d(
    shape: tuple[int, int],
    scale: float = 100.0,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    base_seed: int = 0,
    noise_type: str = 'perlin', # 'perlin' or 'simplex'
    normalize_range: tuple[float, float] = (0.0, 1.0) # Target output range
) -> np.ndarray:
    """
    Generates a 2D procedural noise map (Perlin or Simplex).

    Args:
        shape: Tuple (rows, cols) of the output array.
        scale: Controls the base frequency/zoom level. Higher scale = more detail/higher freq.
               Think of it as pixels per noise cycle (approx). Lower scale = larger features.
        octaves: Number of noise layers summed together. Higher = more detail.
        persistence: Amplitude multiplier for each subsequent octave (usually <= 0.5).
        lacunarity: Frequency multiplier for each subsequent octave (usually >= 2.0).
        base_seed: An integer offset for the noise function to generate different patterns.
        noise_type: Either 'perlin' (pnoise2) or 'simplex' (snoise2).
        normalize_range: Tuple (min, max) for the output noise range. Default [0, 1].

    Returns:
        np.ndarray: 2D noise map of the specified shape and type, normalized to normalize_range.
    """
    rows, cols = shape
    noise_map = np.zeros(shape, dtype=np.float32)

    # Select the noise function from the library
    if noise_type.lower() == 'simplex':
        noise_func = noise.snoise2
    elif noise_type.lower() == 'perlin':
        noise_func = noise.pnoise2
        # due to a known issue [https://github.com/caseman/noise/issues/38] need to limit the seed value to <22,000
        base_seed = int(hashlib.sha256(b'1234567890').hexdigest(), 16) % 22000
    else:
        print(f"Warning: Unknown noise_type '{noise_type}'. Defaulting to Perlin.")
        noise_func = noise.pnoise2
        base_seed = int(hashlib.sha256(b'1234567890').hexdigest(), 16) % 22000
        
    print(f"DEBUG NoiseUtil: Generating {noise_type} noise map ({rows}x{cols}), scale={scale}, octaves={octaves}, seed={base_seed}")
    # Generate noise value for each pixel
    # The noise functions require coordinates scaled appropriately
    for i in range(rows):
        for j in range(cols):
            # Calculate coordinates in noise space
            # Dividing by scale makes features larger for lower scale values
            nx = j / scale
            ny = i / scale

            try:
                noise_map[i][j] = noise_func(
                    nx, ny,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    base=base_seed
                )
            except Exception as noise_err:
                # Catch potential errors from noise lib explicitly
                print(f"ERROR inside noise_func call at ({i},{j}): {noise_err}")
                noise_map[i][j] = 0 # Assign default value on error
            # --- End Potential Crash Point ---

    # Normalize the result
    # Output of pnoise/snoise is roughly in [-1, 1], but not strictly guaranteed.
    # Empirical scaling often works well, or clamp and scale.
    min_val, max_val = normalize_range
    if min_val == -1.0 and max_val == 1.0:
         # Assume output is roughly [-1, 1], maybe clamp just in case
         noise_map = np.clip(noise_map, -1.0, 1.0)
         print(f"DEBUG NoiseUtil: Raw noise range approx [{np.min(noise_map):.2f}, {np.max(noise_map):.2f}], normalizing to [-1, 1]")
    else: # Default normalize to [0, 1]
         # Normalize from approx [-1, 1] to [0, 1]
         noise_map = (noise_map + 1.0) * 0.5
         noise_map = np.clip(noise_map, 0.0, 1.0) # Clip to ensure range
         # If target range is different from [0, 1], scale/shift
         if min_val != 0.0 or max_val != 1.0:
              noise_map = noise_map * (max_val - min_val) + min_val
         print(f"DEBUG NoiseUtil: Raw noise range approx [{np.min(noise_map)*2-1:.2f}, {np.max(noise_map)*2-1:.2f}], normalizing to [{min_val}, {max_val}]")


    print(f"DEBUG NoiseUtil: Final noise map range [{np.min(noise_map):.3f}, {np.max(noise_map):.3f}]")
    return noise_map.astype(np.float32)