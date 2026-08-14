r"""
import_ta_roster_from_class_indices.py

Membuat / melengkapi database roster mahasiswa kelas TA dari:
    models/class_indices_ta.json

Tujuan:
    Supaya Session Report tidak menolak hasil recognition dengan alasan:
    "Dikenali tapi tidak terdaftar di kelas ini"

Dipakai di Raspberry Pi:
    cd <project-root>/src
    python3 import_ta_roster_from_class_indices.py

File yang dibaca:
    <project-root>/models/class_indices_ta.json

File yang ditulis:
    <project-root>/database/mahasiswa_gui.csv

Catatan:
    - Script tidak mengubah model.
    - Script tidak mengubah anti-spoofing.
    - Script hanya menambahkan roster kelas TA.
    - Jika NIM sudah terdaftar pada kelas TA, data dilewati agar tidak duplikat.
"""

from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import re
import shutil
import sys



def get_project_root() -> Path:
    try:
        current_file = Path(__file__).resolve()
        if current_file.parent.name == "src":
            return current_file.parents[1]
    except Exception:
        pass

    return Path.cwd().parent


def sanitize_name(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_label_to_nim_name(label: str):
    """
    Contoh label:
        STUDENT001_Sample_Student
        STUDENT001 Sample Student
        Sample Student

    Output:
        nim, nama
    """
    label = str(label or "").strip()

    if not label:
        return "", ""

    # Pattern paling umum: NIM_Nama atau NIM Nama
    if "_" in label:
        first, rest = label.split("_", 1)
        nim = first.strip()
        nama = sanitize_name(rest)
        return nim, nama if nama else label

    parts = label.split(" ", 1)
    if len(parts) == 2 and re.match(r"^[A-Za-z0-9]+$", parts[0]):
        nim = parts[0].strip()
        nama = sanitize_name(parts[1])
        return nim, nama if nama else label

    # Jika tidak ada NIM, pakai label sebagai NIM dan nama.
    return label, sanitize_name(label)


def read_existing_rows(csv_path: Path):
    if not csv_path.exists():
        return [], ["kelas", "nim", "nama", "folder", "created_at"]

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or ["kelas", "nim", "nama", "folder", "created_at"]
        rows = list(reader)

    required = ["kelas", "nim", "nama", "folder", "created_at"]
    for field in required:
        if field not in fieldnames:
            fieldnames.append(field)

    return rows, fieldnames


def backup_file(path: Path):
    if not path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}_backup_before_import_ta_{timestamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    return backup_path


def main():
    parser = argparse.ArgumentParser(description="Import TA roster from class_indices_ta.json.")
    parser.add_argument("--project_root", type=str, default=str(get_project_root()))
    parser.add_argument("--class_name", type=str, default="TA")
    parser.add_argument("--class_indices", type=str, default="")
    parser.add_argument("--student_csv", type=str, default="")
    parser.add_argument("--create_dataset_folders", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    class_name = args.class_name.strip()

    class_indices_path = Path(args.class_indices) if args.class_indices else project_root / "models" / "class_indices_ta.json"
    student_csv = Path(args.student_csv) if args.student_csv else project_root / "database" / "mahasiswa_gui.csv"
    dataset_class_dir = project_root / "dataset" / "recognition" / class_name

    print("=" * 90)
    print("IMPORT TA ROSTER FROM CLASS INDICES")
    print("=" * 90)
    print(f"Project root  : {project_root}")
    print(f"Class indices : {class_indices_path}")
    print(f"Student CSV   : {student_csv}")
    print(f"Class name    : {class_name}")
    print("=" * 90)

    if not class_indices_path.exists():
        print("\nERROR:")
        print(f"File class_indices_ta.json tidak ditemukan: {class_indices_path}")
        print("Pastikan model TA sudah dipindahkan ke folder models Raspberry Pi.")
        sys.exit(1)

    student_csv.parent.mkdir(parents=True, exist_ok=True)

    with class_indices_path.open("r", encoding="utf-8") as f:
        class_indices = json.load(f)

    labels = sorted(class_indices.keys(), key=lambda k: int(class_indices[k]))

    rows, fieldnames = read_existing_rows(student_csv)

    existing_keys = set()
    for row in rows:
        kelas = str(row.get("kelas", "")).strip()
        nim = str(row.get("nim", "")).strip()
        if kelas == class_name and nim:
            existing_keys.add((kelas, nim))

    backup_path = backup_file(student_csv)

    added = 0
    skipped = 0

    for label in labels:
        nim, nama = split_label_to_nim_name(label)

        if not nim:
            skipped += 1
            continue

        key = (class_name, nim)

        if key in existing_keys:
            skipped += 1
            continue

        dataset_folder = dataset_class_dir / label

        if args.create_dataset_folders:
            dataset_folder.mkdir(parents=True, exist_ok=True)

        new_row = {
            "kelas": class_name,
            "nim": nim,
            "nama": nama,
            "folder": str(dataset_folder.relative_to(project_root)),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        for field in fieldnames:
            new_row.setdefault(field, "")

        rows.append(new_row)
        existing_keys.add(key)
        added += 1

    with student_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("\nSUMMARY")
    print("=" * 90)
    print(f"Labels in class_indices : {len(labels)}")
    print(f"Added TA roster rows    : {added}")
    print(f"Skipped existing rows   : {skipped}")
    print(f"Total CSV rows now      : {len(rows)}")
    if backup_path:
        print(f"Backup created          : {backup_path}")
    else:
        print("Backup created          : none, CSV did not exist before")
    print("=" * 90)
    print("Selesai.")
    print("Tutup GUI lalu buka ulang agar Data Registration dan Session Report membaca database terbaru.")


if __name__ == "__main__":
    main()
