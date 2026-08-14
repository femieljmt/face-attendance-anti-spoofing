# Model Files

The original package contained trained face-recognition models and class-index mappings tied to identifiable students. Those files are intentionally excluded from this public-ready package.

## Expected private model files

Place your working model files locally in this directory before running recognition:

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

`*.h5` and real `class_indices_*.json` files are ignored by Git in this package.

`class_indices.example.json` demonstrates the expected JSON structure using fictional identities.

The anti-spoofing threshold JSON is included because the supplied file contained no student identity data. Model licensing and dataset provenance should still be verified before publishing any trained model binary.
