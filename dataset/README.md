# Dataset Pengenalan Wajah

Foto wajah yang digunakan saat pengembangan tidak dipublikasikan pada repository ini. Dataset harus disiapkan sendiri pada komputer yang menjalankan proyek.

Susunan folder yang dibaca aplikasi:

```text
dataset/
└── recognition/
    ├── PCD/
    ├── TA/
    ├── Pempros/
    └── PSD/
```

Di dalam setiap kelas, satu identitas dapat ditempatkan pada folder yang labelnya sesuai dengan pemetaan model. Contoh dengan identitas fiktif:

```text
dataset/recognition/PCD/STUDENT001_Contoh_Mahasiswa/
```

Nama folder dan `class_indices` harus konsisten dengan model recognition yang digunakan. Jangan mempublikasikan dataset wajah tanpa izin peserta dan dasar penggunaan data yang sesuai.
