import sys
import pymongo
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, PyMongoError
from backend.config import settings

_db = None
_client = None

import urllib.parse

def clean_mongodb_uri(uri: str) -> str:
    """Escapes username and password inside a MongoDB URI according to RFC 3986."""
    if not uri.startswith("mongodb://") and not uri.startswith("mongodb+srv://"):
        return uri
    try:
        prefix = "mongodb+srv://" if uri.startswith("mongodb+srv://") else "mongodb://"
        rest = uri[len(prefix):]
        if "@" not in rest:
            return uri
        auth_part, host_part = rest.rsplit("@", 1)
        if ":" not in auth_part:
            return uri
        username, password = auth_part.split(":", 1)
        escaped_username = urllib.parse.quote_plus(username)
        escaped_password = urllib.parse.quote_plus(password)
        return f"{prefix}{escaped_username}:{escaped_password}@{host_part}"
    except (ValueError, IndexError, AttributeError):
        return uri

def get_database():
    global _db, _client
    if _db is not None:
        return _db
    
    try:
        cleaned_uri = clean_mongodb_uri(settings.mongodb_uri)
        # Initialize client with 10000ms timeout to avoid spurious connection failures during cold starts
        _client = pymongo.MongoClient(cleaned_uri, serverSelectionTimeoutMS=10000)
        # Force a connection check by pinging the admin database
        _client.admin.command('ping')
        _db = _client[settings.mongodb_database]
        return _db
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print("\n" + "="*80, file=sys.stderr)
        print("ERROR: MongoDB is unavailable. Please verify the configured MongoDB connection.", file=sys.stderr)
        print("\nSetup Instructions:", file=sys.stderr)
        print("1. Ensure MongoDB is installed and running locally, or verify your MongoDB Atlas credentials.", file=sys.stderr)
        print("2. Set the MONGODB_URI in backend/.env to your correct MongoDB connection string.", file=sys.stderr)
        print("   Example: MONGODB_URI=mongodb://localhost:27017 or MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net", file=sys.stderr)
        print("3. Make sure the database service is started (e.g., 'net start MongoDB' on Windows or 'sudo systemctl start mongod' on Linux).", file=sys.stderr)
        print("="*80 + "\n", file=sys.stderr)
        raise RuntimeError("MongoDB connection unavailable. Please verify the configured MongoDB connection.") from e

def init_db_indexes():
    try:
        db = get_database()
        
        # Build required indexes
        db.cases.create_index("case_id", unique=True)
        db.cases.create_index("created_at")
        db.cases.create_index("model_version")
        
        db.predictions.create_index("case_id")
        db.predictions.create_index("model_version")
        
        db.explanations.create_index("case_id")
        db.reports.create_index("case_id")
        
        db.model_versions.create_index([("model_name", 1), ("model_version", 1)], unique=True)
        
        print("MongoDB collection indexes initialized successfully.")
    except PyMongoError as e:
        print(f"Failed to initialize database indexes: {str(e)}", file=sys.stderr)
        # Don't crash here since database retrieval itself handles the critical failure,
        # but report index building issues.
