"""
Course submission entry point.

Delegates entirely to app/main.py — no logic lives here.
Run with:  streamlit run streamlit_app.py
"""

import sys
import os
import runpy

# Make `from utils.xxx import` work (same as app/main.py does via sys.path.insert)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))

# Execute app/main.py as __main__ — identical to running it directly
runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "main.py"),
    run_name="__main__",
)
