import os
import json
import numpy as np
import imageio
import cv2 # Using OpenCV for image saving initially
# Consider adding 'import tifffile' for better TIF handling if needed later
from typing import Dict, Any, Union, List

def save_gif(frames: List[np.ndarray], filepath: str, fps: int = 5, loop: int = 0):
    """
    Saves a list of numpy arrays (frames) as an animated GIF.

    Args:
        frames (List[np.ndarray]): List of frames (uint8). Assumes grayscale or BGR.
        filepath (str): Path to save the GIF file.
        fps (int): Frames per second for the GIF.
        loop (int): Number of times the GIF should loop (0 = infinite).
    """
    if not frames:
        print("Warning: No frames provided for GIF generation.")
        return
    ensure_dir_exists(os.path.dirname(filepath))
    try:
        # Ensure frames are uint8
        uint8_frames = []
        for frame in frames:
            if frame is None: continue
            if frame.dtype != np.uint8:
                 # Attempt conversion, assuming float [0,1] or scaling needed
                 if np.issubdtype(frame.dtype, np.floating):
                     frame = scale_to_uint(frame, np.uint8)
                 elif np.issubdtype(frame.dtype, np.integer):
                     # Try direct cast, might be wrong range
                     frame = frame.astype(np.uint8)
                 else:
                     print(f"Warning: Skipping frame with unsupported dtype {frame.dtype} for GIF.")
                     continue
            uint8_frames.append(frame)

        if not uint8_frames:
             print("Warning: No valid uint8 frames found for GIF generation.")
             return

        print(f"DEBUG IO: Saving GIF with {len(uint8_frames)} frames to {filepath}")
        # Use imageio v3 API if available, otherwise fallback
        try:
             imageio.mimsave(filepath, uint8_frames, duration=(1000 // fps), loop=loop) # duration in ms
        except AttributeError: # Fallback for older imageio v2?
             imageio.mimsave(filepath, uint8_frames, fps=fps, loop=loop)

    except Exception as e:
        print(f"Error saving GIF {filepath}: {e}")

def ensure_dir_exists(path: str):
     """Creates a directory if it doesn't exist."""
     # print(f"DEBUG: ensure_dir_exists attempting path: '{path}'") # Add this line
     if not path: # Explicit check for empty path
          print("ERROR: ensure_dir_exists received an empty path!")
          raise ValueError("Attempted to create directory with an empty path.")
     try:
         os.makedirs(path, exist_ok=True)

     except OSError as e:
         print(f"Error creating directory {path}: {e}")
         raise

def scale_to_uint(img: np.ndarray, dtype: Union[np.uint8, np.uint16]) -> np.ndarray:
     """Scales float image [0, 1] to uint8/uint16 range."""
     if not np.issubdtype(img.dtype, np.floating):
         # If already int, assume correct range or return as is
         if img.dtype == dtype: return img
         print(f"Warning: scale_to_uint received non-float image {img.dtype}. Trying to convert.")
         return img.astype(dtype) # Might be incorrect range

     if dtype == np.uint8:
         max_val = 255
     elif dtype == np.uint16:
         max_val = 65535
     else:
         raise ValueError("Unsupported dtype for scaling. Use uint8 or uint16.")

     scaled_img = (np.clip(img, 0.0, 1.0) * max_val).astype(dtype)
     return scaled_img

def save_image(img: np.ndarray, filepath: str):
    """Saves numpy array as an image file (TIF or PNG)."""
    ensure_dir_exists(os.path.dirname(filepath))
    ext = os.path.splitext(filepath)[1].lower()

    try:
        save_data = img
        # Convert float [0,1] images to appropriate integer range before saving
        if np.issubdtype(img.dtype, np.floating):
            if ext == ".png": # PNG usually expects uint8 or uint16
                # Decide based on expected bit depth or just default to uint8/16
                # Let's assume 16-bit if values suggest high dynamic range, else 8-bit?
                # Safer to default based on config's target bit depth if known, else uint16
                save_data = scale_to_uint(img, np.uint16)
            elif ext == ".tif":
                # TIF can store float, but often uint16 is preferred for compatibility
                save_data = scale_to_uint(img, np.uint16) # Save TIF as uint16
        elif np.issubdtype(img.dtype, np.integer):
             # Assume integer images are already in correct range
             pass

        # Use OpenCV for saving (simpler, less control than tifffile)
        # For TIF, use compression params if desired
        params = []
        if ext == ".tif":
             # Add compression? e.g., LZW
             # params = [cv2.IMWRITE_TIFF_COMPRESSION, 5] # 5 is LZW
             pass # No compression by default

        success = cv2.imwrite(filepath, save_data, params=params)
        if not success:
             print(f"Warning: cv2.imwrite failed for {filepath}")

    except Exception as e:
        print(f"Error saving image {filepath}: {e}")

def save_metadata(metadata: Dict[str, Any], filepath: str):
    """Saves metadata dictionary as a JSON file."""
    ensure_dir_exists(os.path.dirname(filepath))
    try:
        with open(filepath, 'w') as f:
            # Use default=str to handle non-serializable types gracefully (like numpy types)
            json.dump(metadata, f, indent=4, default=str)
    except Exception as e:
        print(f"Error saving metadata {filepath}: {e}")

def save_text(text_content: str, filepath: str):
    """Saves string content to a text file."""
    ensure_dir_exists(os.path.dirname(filepath))
    try:
        with open(filepath, 'w') as f:
            f.write(text_content)
    except Exception as e:
        print(f"Error saving text file {filepath}: {e}")