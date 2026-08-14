"""Convenience entry point for the desktop attendance application."""
from pathlib import Path
import runpy

APP = Path(__file__).resolve().parent / "src" / "attendance_gui_tkinter_session_v8_best_confidence.py"
runpy.run_path(str(APP), run_name="__main__")
