"""
Main launcher for WhatsApp Chat Analyzer
Run this from the project root directory
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now import and run the app
from app import create_streamlit_app

if __name__ == "__main__":
    create_streamlit_app()