import os
import sys

# Add root directory to python path for Vercel Serverless Environment
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Set Vercel environment flag
os.environ["VERCEL"] = "1"

# Import FastAPI application
from backend.main import app

# Export handler for Vercel
handler = app
