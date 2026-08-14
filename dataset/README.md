# Recognition Dataset

Real face images are intentionally excluded from this public-ready package.

At runtime the project expects recognition data under:

```text
dataset/
└── recognition/
    ├── PCD/
    ├── TA/
    ├── Pempros/
    └── PSD/
```

Each identity may be represented by a subdirectory matching the label used by the recognition model, for example:

```text
dataset/recognition/PCD/STUDENT001_Sample_Student/
```

Do not publish face datasets without an appropriate legal basis and participant permission.
