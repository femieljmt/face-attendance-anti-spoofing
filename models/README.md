# Required Models

The original face-recognition models and class-index files are not included because they are connected to student identities. Copy the project's private model files into this directory before running recognition.

The application expects these filenames:

```text
models/
├── anti_spoofing_mobilenet_real_spoof.h5
├── anti_spoofing_threshold.json
├── face_recognition_mobilenet_pcd.h5
├── face_recognition_mobilenet_ta.h5
├── face_recognition_mobilenet_pempros.h5
├── face_recognition_mobilenet_psd.h5
├── class_indices_pcd.json
├── class_indices_ta.json
├── class_indices_pempros.json
└── class_indices_psd.json
```

The original `*.h5` and `class_indices_*.json` files are excluded by `.gitignore`. `class_indices.example.json` shows the expected JSON structure with fictional identities.

`anti_spoofing_threshold.json` remains in the repository because the supplied file does not contain student identities. Before publishing any other trained model, verify the dataset source, permission to release it, and its license terms.
