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

class MockCollection:
    def __init__(self, name):
        self.name = name
        self._store = {}

    def create_index(self, *args, **kwargs):
        pass

    def find_one(self, query=None, projection=None):
        if not query:
            return next(iter(self._store.values()), None)
        case_id = query.get("case_id")
        if case_id and case_id in self._store:
            return dict(self._store[case_id])
        for doc in self._store.values():
            if all(doc.get(k) == v for k, v in query.items()):
                return dict(doc)
        return None

    def insert_one(self, doc):
        case_id = doc.get("case_id") or str(len(self._store) + 1)
        self._store[case_id] = dict(doc)
        class MockResult:
            inserted_id = case_id
        return MockResult()

    def update_one(self, filter_query, update_doc, upsert=False):
        doc = self.find_one(filter_query)
        if doc:
            cid = doc.get("case_id")
            if "$set" in update_doc and cid in self._store:
                self._store[cid].update(update_doc["$set"])
        elif upsert:
            new_doc = dict(filter_query)
            if "$set" in update_doc:
                new_doc.update(update_doc["$set"])
            self.insert_one(new_doc)

    def find(self, query=None, projection=None):
        docs = [dict(d) for d in self._store.values()]
        class MockCursor(list):
            def sort(self, *args, **kwargs):
                return self
            def limit(self, n):
                return MockCursor(self[:n])
        return MockCursor(docs)

    def count_documents(self, query=None):
        return len(self._store)

class MockMongoDatabase:
    def __init__(self):
        self._collections = {}

    def __getitem__(self, name):
        if name not in self._collections:
            self._collections[name] = MockCollection(name)
        return self._collections[name]

    def __getattr__(self, name):
        return self[name]


def get_database():
    global _db, _client
    if _db is not None:
        return _db
    
    try:
        cleaned_uri = clean_mongodb_uri(settings.mongodb_uri)
        # Initialize client with 5000ms timeout to avoid spurious connection failures during cold starts
        _client = pymongo.MongoClient(cleaned_uri, serverSelectionTimeoutMS=5000)
        # Force a connection check by pinging the admin database
        _client.admin.command('ping')
        _db = _client[settings.mongodb_database]
        return _db
    except (ConnectionFailure, ServerSelectionTimeoutError, PyMongoError) as e:
        if os.environ.get("VERCEL") or os.environ.get("SERVERLESS") or os.environ.get("ALLOW_MOCK_DB"):
            print("WARNING: MongoDB connection unavailable on Vercel environment. Operating with in-memory database fallback.", file=sys.stderr)
            _db = MockMongoDatabase()
            return _db
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
