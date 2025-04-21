import sys
import os

# Add the 'src' directory to the Python path
# This allows absolute imports from 'semgen' assuming main.py is in the root
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now use absolute imports starting from 'semgen'
from semgen.interfaces.cli import run_cli
# Import WebUI app creator if needed, potentially check arguments to decide mode

def main():
    # Basic check: if specific args for webui are present, run webui, else run cli
    # Or just always run CLI for now. Web UI needs separate runner usually.
    # Example: if '--webui' in sys.argv: run_webui() else: run_cli()
    print("Running SEM Generator CLI...")
    run_cli()

if __name__ == "__main__":
    main()

# To run the Web UI separately (e.g., using flask run or gunicorn):
# You would typically have a separate entry point or use environment variables
# Example (in a separate run_web.py or using Flask's command):
#from src.semgen.interfaces.webui.app import create_app # Need to adjust path here too
#app = create_app()
#if __name__ == '__main__':
#    app.run(debug=True) # Debug mode for development
