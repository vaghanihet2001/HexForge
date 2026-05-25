import os
from flask import Flask
from pymongo import MongoClient

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
