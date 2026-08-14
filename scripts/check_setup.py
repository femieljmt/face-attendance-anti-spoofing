"""Check whether the local runtime files expected by the application are present."""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]

DEPENDENCIES = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "tensorflow": "tensorflow",
    "openpyxl": "openpyxl",
}

REQUIRED_MODELS = [
    "anti_spoofing_mobilenet_real_spoof.h5",
    "anti_spoofing_threshold.json",
    "face_recognition_mobilenet_pcd.h5",
    "face_recognition_mobilenet_ta.h5",
    "face_recognition_mobilenet_pempros.h5",
    "face_recognition_mobilenet_psd.h5",
    "class_indices_pcd.json",
    "class_indices_ta.json",
    "class_indices_pempros.json",
    "class_indices_psd.json",
]

print(f"Project root: {ROOT}")
print("\nPython dependencies:")
missing_deps = []
for module, package in DEPENDENCIES.items():
    ok = importlib.util.find_spec(module) is not None
    print(f"  [{'OK' if ok else 'MISSING'}] {package}")
    if not ok:
        missing_deps.append(package)

print("\nLocal model files:")
missing_models = []
for name in REQUIRED_MODELS:
    ok = (ROOT / "models" / name).exists()
    print(f"  [{'OK' if ok else 'MISSING'}] models/{name}")
    if not ok:
        missing_models.append(name)

print("\nSummary:")
if not missing_deps and not missing_models:
    print("  Setup appears complete for the supplied application structure.")
else:
    if missing_deps:
        print("  Install missing dependencies with: python -m pip install -r requirements.txt")
    if missing_models:
        print("  Copy your private trained model files into models/. See models/README.md.")
