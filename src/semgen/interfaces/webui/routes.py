from flask import (
    Blueprint, render_template, request, jsonify, current_app, send_from_directory
)
import json
import os
import time
import random
import threading
import tempfile
import shutil

from ...config_loader import load_config, DEFAULT_CONFIG, merge_configs
from ...pipeline import generate_sample # Import the core generation logic

# Use a Blueprint for better organization
bp = Blueprint('main', __name__, url_prefix='/')

# --- State Management (Simple In-Memory - Replace for Production) ---
# WARNING: This is NOT suitable for production use with multiple users!
# Use proper task queues (Celery, RQ) and databases for real applications.
generation_status = {
    "running": False,
    "progress": 0,
    "total": 0,
    "message": "Idle",
    "last_output_zip": None,
}
generation_thread = None

# --- Helper Function ---
def run_generation_in_background(config):
    global generation_status, generation_thread
    try:
        num_samples_to_gen = config['num_samples']
        master_seed = config['seed']
        # Create a temporary directory for this batch
        temp_batch_dir = tempfile.mkdtemp(prefix="semgen_batch_")
        generation_status["message"] = f"Starting generation in {os.path.basename(temp_batch_dir)}..."
        generation_status["progress"] = 0
        generation_status["total"] = num_samples_to_gen
        generation_status["last_output_zip"] = None

        print(f"Background thread: Generating {num_samples_to_gen} samples to {temp_batch_dir}")

        for i in range(num_samples_to_gen):
            if not generation_status["running"]: # Check if cancelled
                 generation_status["message"] = "Generation cancelled."
                 break
            generation_status["message"] = f"Generating sample {i+1}/{num_samples_to_gen}..."
            generation_status["progress"] = i + 1
            try:
                # generate_sample creates its own subfolder within the passed dir
                generate_sample(config, i, temp_batch_dir, master_seed)
            except Exception as e:
                print(f"Error in background generation sample {i}: {e}")
                generation_status["message"] = f"Error during sample {i+1}: {e}"
                # Decide whether to stop or continue
                # break # Stop on first error for now
        else: # Only runs if loop completes without break
             generation_status["message"] = "Generation complete. Compressing output..."
             # --- Zip the output ---
             if os.path.exists(temp_batch_dir) and any(os.scandir(temp_batch_dir)):
                  try:
                       zip_filename_base = f"semgen_output_{time.strftime('%Y%m%d_%H%M%S')}"
                       # Save zip outside temp dir, maybe in instance_path or a dedicated output dir?
                       # Saving in instance path for simplicity here
                       zip_output_path = os.path.join(current_app.instance_path, zip_filename_base)
                       print(f"Zipping output to {zip_output_path}.zip")
                       shutil.make_archive(zip_output_path, 'zip', temp_batch_dir)
                       generation_status["last_output_zip"] = f"{zip_filename_base}.zip"
                       generation_status["message"] = "Output zipped successfully."
                       print("Zipping complete.")
                  except Exception as e:
                       print(f"Error zipping output: {e}")
                       generation_status["message"] = f"Error zipping output: {e}"
             else:
                 generation_status["message"] = "Generation finished, but no output found to zip."


    except Exception as e:
        print(f"Error in background generation thread: {e}")
        generation_status["message"] = f"Error: {e}"
    finally:
        generation_status["running"] = False
        generation_status["progress"] = 0 # Reset progress
        # Clean up temporary directory
        if temp_batch_dir and os.path.exists(temp_batch_dir):
            try:
                print(f"Cleaning up temporary directory: {temp_batch_dir}")
                shutil.rmtree(temp_batch_dir)
            except Exception as e:
                print(f"Error cleaning up temp directory {temp_batch_dir}: {e}")
        print("Background generation thread finished.")


# --- Routes ---
@bp.route('/')
def index():
    """Main page displaying the configuration UI."""
    # Pass default config structure to template for rendering controls
    # In a real app, you'd use forms (like WTForms)
    return render_template('index.html', default_config=DEFAULT_CONFIG)

@bp.route('/start_generation', methods=['POST'])
def start_generation():
    """Receives config from UI and starts background generation."""
    global generation_status, generation_thread
    if generation_status["running"]:
        return jsonify({"status": "error", "message": "Generation already in progress."}), 400

    try:
        # Get config from POST request (assume JSON body)
        user_config = request.json
        if not user_config:
             return jsonify({"status": "error", "message": "No configuration data received."}), 400

        # Merge with defaults and validate basic structure (more needed)
        config = merge_configs(DEFAULT_CONFIG, user_config)

        # Ensure seed exists
        if config.get('seed') is None or config.get('seed') == 0:
             config['seed'] = random.randint(1, 2**32 - 1)

        # Start generation in a background thread
        generation_status["running"] = True
        generation_status["message"] = "Initializing..."
        generation_thread = threading.Thread(target=run_generation_in_background, args=(config.copy(),)) # Pass copy
        generation_thread.start()

        return jsonify({"status": "ok", "message": "Generation started."})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to start generation: {e}"}), 500

@bp.route('/get_status')
def get_status():
    """Endpoint for the frontend to poll generation status."""
    return jsonify(generation_status)

@bp.route('/cancel_generation', methods=['POST'])
def cancel_generation():
    """Attempts to stop the background generation."""
    global generation_status, generation_thread
    if not generation_status["running"]:
        return jsonify({"status": "ok", "message": "No generation running."})

    print("Cancellation requested.")
    generation_status["running"] = False # Signal thread to stop
    # Note: Thread might take time to finish current sample
    # For forceful stop, more complex mechanisms are needed.

    return jsonify({"status": "ok", "message": "Cancellation signal sent."})

@bp.route('/download_output')
def download_output():
    """Allows downloading the generated zip file."""
    zip_filename = generation_status.get("last_output_zip")
    if not zip_filename:
        return "No output file available.", 404

    download_dir = current_app.instance_path
    filepath = os.path.join(download_dir, zip_filename)

    if not os.path.exists(filepath):
         return f"Output file {zip_filename} not found.", 404

    # TODO: Implement delete_on_download logic if needed
    # Requires careful handling, maybe move file after send_from_directory?

    print(f"Sending file: {filepath}")
    try:
        return send_from_directory(download_dir, zip_filename, as_attachment=True)
    except Exception as e:
        print(f"Error sending file: {e}")
        return "Error sending file.", 500