from flask import Flask
import os

# Import routes after app is created to avoid circular imports
# from . import routes

def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__, instance_relative_config=True)

    # Basic configuration (can be expanded)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-secret-key'), # Change for production!
        # Add other configurations like upload folder, etc.
    )

    # Ensure the instance folder exists (for potential uploads, session data)
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        pass

    # Register blueprints or routes here
    # Example: app.register_blueprint(routes.bp)
    # For simplicity now, define routes directly or import later
    from . import routes
    app.register_blueprint(routes.bp)


    print("Flask app created.")
    print(f"Templates folder: {app.template_folder}")
    print(f"Static folder: {app.static_folder}")


    return app