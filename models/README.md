# Model yang Dibutuhkan

Model pengenalan wajah dan file indeks kelas asli tidak dimasukkan ke repository karena terhubung dengan identitas mahasiswa. Sebelum menjalankan recognition, salin model milik proyek ke folder ini.

Nama file yang dibaca aplikasi:

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

File `*.h5` dan `class_indices_*.json` asli dikecualikan melalui `.gitignore`. File `class_indices.example.json` hanya menunjukkan format JSON menggunakan identitas fiktif.

`anti_spoofing_threshold.json` tetap disertakan karena file yang tersedia tidak memuat identitas mahasiswa. Sebelum model biner lain dipublikasikan, pastikan asal dataset, izin penggunaan, dan ketentuan lisensinya sudah jelas.
