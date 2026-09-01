import urllib.parse
from backend.config import settings

uri = settings.mongodb_uri
print("Raw URI in settings:", uri)

def clean_mongodb_uri(uri: str) -> str:
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
        print("Parsed username:", username)
        print("Parsed password:", password)
        escaped_username = urllib.parse.quote_plus(username)
        escaped_password = urllib.parse.quote_plus(password)
        res = f"{prefix}{escaped_username}:{escaped_password}@{host_part}"
        print("Cleaned URI:", res)
        return res
    except Exception as e:
        print("Error during clean:", str(e))
        return uri

cleaned = clean_mongodb_uri(uri)

import pymongo
try:
    client = pymongo.MongoClient(cleaned)
    print("MongoClient created successfully.")
except Exception as e:
    print("MongoClient failed:", str(e))
