import os
from flask import Flask
from dotenv import load_dotenv

load_dotenv()

def create_app(test_config=None):
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True,
                template_folder='../templates',
                static_folder='../static')
    
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev'),
        UPLOAD_FOLDER=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/uploads'),
        PROCESSED_FOLDER=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/processed'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max upload size
        ALLOWED_EXTENSIONS={'csv', 'xls', 'xlsx', 'json', 'txt'}
    )

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)

    # Ensure the upload and processed folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

    # Register blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
