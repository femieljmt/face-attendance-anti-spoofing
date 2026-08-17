# Face-Recognition Dataset

The face images used during development are not published in this repository. Prepare the dataset locally on the computer that runs the project.

The application expects the following directory layout:

```text
dataset/
└── recognition/
    ├── PCD/
    ├── TA/
    ├── Pempros/
    └── PSD/
```

Within each class, an identity can be stored in a folder whose label matches the recognition model's class mapping. For example, using a fictional identity:

```text
dataset/recognition/PCD/STUDENT001_Sample_Student/
```

The folder labels and `class_indices` files must be consistent with the recognition model. Do not publish face datasets without participant consent and an appropriate basis for using the data.
