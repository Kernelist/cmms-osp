"""
Vercel entry point — re-exports the FastAPI app from api/main.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from api.main import app  # noqa: F401  (Vercel looks for `app`)
