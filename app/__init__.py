import os
import logging
from flask import Flask
from pymongo import MongoClient

# ── Logging setup ─────────────────────────────────────────────────────────────
import logging.handlers

_LOG_DIR  = os.path.join(os.path.dirname(__file__), '..', 'logs')
_LOG_FILE = os.path.join(_LOG_DIR, 'hexforge.log')
os.makedirs(_LOG_DIR, exist_ok=True)

_log_fmt = logging.Formatter(
    '[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Console handler
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_log_fmt)

# Rotating file handler — 5 MB per file, keep last 3 files
_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8'
)
_file_handler.setFormatter(_log_fmt)

_hexforge_logger = logging.getLogger('hexforge')
_hexforge_logger.setLevel(logging.DEBUG)
_hexforge_logger.addHandler(_stream_handler)
_hexforge_logger.addHandler(_file_handler)
_hexforge_logger.propagate = False   # don't double-log via root logger
# ─────────────────────────────────────────────────────────────────────────────

# Initialize PyMongo client
mongo_client = None
db = None

def create_app():
    global mongo_client, db
    
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'hexforge_dev_key')
    app.config['MONGO_URI'] = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
    
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize MongoDB
    try:
        mongo_client = MongoClient(app.config['MONGO_URI'], serverSelectionTimeoutMS=5000)
        # Verify connection
        mongo_client.server_info()
        db = mongo_client['hexforge']
        print("Successfully connected to MongoDB 'hexforge' database.")
    except Exception as e:
        print(f"Warning: Could not connect to MongoDB. {e}")
        db = None

    # Register Blueprints
    from app.routes.main_routes import main_bp
    from app.routes.api_routes import api_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)

    return app
