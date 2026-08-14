r"""
repair_raspi_dataset_recognition_and_roster.py

Memperbaiki struktur folder dataset/recognition dan roster GUI pada Raspberry Pi.

Masalah yang diperbaiki:
    - Folder dataset/recognition/TA tidak ada.
    - Folder dataset/recognition/PCD, Pempros, PSD tidak ada.
    - Tombol "Import Students from Dataset Folders" gagal karena folder kelas tidak ditemukan.
    - Hasil recognition ditolak karena mahasiswa belum tercatat di database/mahasiswa_gui.csv.

Script ini TIDAK menghapus model dan TIDAK menghapus database.
Script hanya:
    1. Membuat folder kelas:
        dataset/recognition/PCD
        dataset/recognition/TA
        dataset/recognition/Pempros
        dataset/recognition/PSD

    2. Membaca class_indices dari folder models:
        models/class_indices_pcd.json
        models/class_indices_ta.json
        models/class_indices_pempros.json
        models/class_indices_psd.json

    3. Membuat subfolder mahasiswa berdasarkan label class_indices.

    4. Menambahkan data mahasiswa ke:
        database/mahasiswa_gui.csv

Cara pakai di Raspberry Pi:
    cd <project-root>/src
    python3 repair_raspi_dataset_recognition_and_roster.py

Jika hanya ingin membuat folder kelas tanpa import roster:
    python3 repair_raspi_dataset_recognition_and_roster.py --folders_only
"""

from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import re
import shutil
import sys


CLASSES = ["PCD", "TA", "Pempros", "PSD"]

def get_project_root() -> Path:
    try:
        here = Path(__file__).resolve()
        if here.parent.name == "src":
            return here.parents[1]
    except Exception:
        pass

    return Path.cwd().parent


def sanitize_folder_name(text: str) -> str:
    text = str(text or "").strip()
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ["_", "-"]:
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    return "".join(keep).strip("_")


def clean_name(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_label_to_nim_name(label: str):
    label = str(label or "").strip()

    if not label:
        return "", ""

    # Format umum: NIM_Nama
    if "_" in label:
        first, rest = label.split("_", 1)
        nim = first.strip()
        nama = clean_name(rest)
        if not nama:
            nama = clean_name(label)
        return nim, nama

    # Format alternatif: NIM Nama
    parts = label.split(" ", 1)
    if len(parts) == 2 and re.match(r"^[A-Za-z0-9]+$", parts[0]):
        nim = parts[0].strip()
        nama = clean_name(parts[1])
        if not nama:
            nama = clean_name(label)
        return nim, nama

    # Kalau label hanya nama, pakai label sebagai NIM dan nama agar tetap bisa dicocokkan.
    return sanitize_folder_name(label), clean_name(label)


def class_to_slug(kelas: str) -> str:
    return kelas.strip().lower()


def class_indices_path(project_root: Path, kelas: str) -> Path:
    return project_root / "models" / f"class_indices_{class_to_slug(kelas)}.json"


def read_class_labels(path: Path):
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # class_indices format: {"label": index}
    try:
        return sorted(data.keys(), key=lambda label: int(data[label]))
    except Exception:
        return sorted(data.keys())


def read_existing_csv(csv_path: Path):
    default_fields = ["kelas", "nim", "nama", "folder", "created_at"]

    if not csv_path.exists():
        return [], default_fields

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or default_fields
        rows = list(reader)

    for field in default_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    return rows, fieldnames


def backup_file(path: Path):
    if not path.exists():
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}_backup_before_repair_dataset_{ts}{path.suffix}")
    shutil.copy2(path, backup_path)
    return backup_path


def make_class_folders(project_root: Path):
    dataset_root = project_root / "dataset" / "recognition"
    dataset_root.mkdir(parents=True, exist_ok=True)

    created_or_exists = []

    for kelas in CLASSES:
        class_dir = dataset_root / kelas
        class_dir.mkdir(parents=True, exist_ok=True)
        created_or_exists.append(class_dir)

    return created_or_exists


def repair_roster(project_root: Path, folders_only=False):
    dataset_root = project_root / "dataset" / "recognition"
    database_dir = project_root / "database"
    student_csv = database_dir / "mahasiswa_gui.csv"

    database_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    class_dirs = make_class_folders(project_root)

    print("=" * 90)
    print("REPAIR DATASET RECOGNITION FOLDERS")
    print("=" * 90)
    for d in class_dirs:
        print(f"[OK] {d}")

    if folders_only:
        print("\nMODE folders_only aktif. Roster CSV tidak diubah.")
        return

    rows, fieldnames = read_existing_csv(student_csv)
    backup_path = backup_file(student_csv)

    existing = set()
    for row in rows:
        kelas = str(row.get("kelas", "")).strip()
        nim = str(row.get("nim", "")).strip()
        if kelas and nim:
            existing.add((kelas, nim))

    added = 0
    skipped_existing = 0
    missing_indices = []
    created_student_folders = 0

    print("\n" + "=" * 90)
    print("IMPORT ROSTER FROM CLASS INDICES")
    print("=" * 90)

    for kelas in CLASSES:
        ci_path = class_indices_path(project_root, kelas)
        class_dir = dataset_root / kelas

        labels = read_class_labels(ci_path)

        if not labels:
            missing_indices.append(ci_path)
            print(f"[SKIP] {kelas}: class_indices tidak ditemukan atau kosong: {ci_path}")
            continue

        print(f"[READ] {kelas}: {len(labels)} label dari {ci_path}")

        for label in labels:
            nim, nama = split_label_to_nim_name(label)

            if not nim:
                continue

            # Folder mahasiswa harus mengikuti label model agar result backend cocok dengan roster.
            student_folder = class_dir / label
            if not student_folder.exists():
                student_folder.mkdir(parents=True, exist_ok=True)
                created_student_folders += 1

            key = (kelas, nim)
            if key in existing:
                skipped_existing += 1
                continue

            row = {
                "kelas": kelas,
                "nim": nim,
                "nama": nama,
                "folder": str(student_folder.relative_to(project_root)),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            for field in fieldnames:
                row.setdefault(field, "")

            rows.append(row)
            existing.add(key)
            added += 1

    with student_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Project root              : {project_root}")
    print(f"Dataset recognition root  : {dataset_root}")
    print(f"Student CSV               : {student_csv}")
    print(f"Added roster rows         : {added}")
    print(f"Skipped existing rows     : {skipped_existing}")
    print(f"Created student folders   : {created_student_folders}")
    print(f"Total CSV rows now        : {len(rows)}")

    if backup_path:
        print(f"Backup CSV                : {backup_path}")
    else:
        print("Backup CSV                : none, CSV did not exist before")

    if missing_indices:
        print("\nPERINGATAN: class_indices berikut tidak ditemukan/kosong:")
        for p in missing_indices:
            print(f"  - {p}")

    print("=" * 90)
    print("Selesai.")
    print("Tutup GUI lalu buka ulang. Setelah itu klik Refresh Tabel atau Import Students from Dataset Folders.")


def main():
    parser = argparse.ArgumentParser(description="Repair Raspi dataset/recognition folders and GUI roster.")
    parser.add_argument("--project_root", type=str, default=str(get_project_root()))
    parser.add_argument("--folders_only", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    if not project_root.exists():
        print(f"ERROR: project_root tidak ditemukan: {project_root}")
        sys.exit(1)

    repair_roster(project_root, folders_only=args.folders_only)


if __name__ == "__main__":
    main()
