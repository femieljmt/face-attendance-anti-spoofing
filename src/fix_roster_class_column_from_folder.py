r"""
fix_roster_class_column_from_folder.py

Memperbaiki kolom "kelas" pada database/mahasiswa_gui.csv.

Masalah:
    Pada tabel Data Registration, kolom paling kiri "Kelas" kosong/tidak menunjukkan
    PCD, TA, Pempros, atau PSD.

Penyebab umum:
    - mahasiswa_gui.csv lama dibuat sebelum struktur kelas diperbaiki.
    - beberapa baris roster punya folder/nama mahasiswa, tetapi kolom kelas kosong.
    - hasil import lama tersimpan tanpa penanda kelas.

Yang dilakukan script:
    1. Backup database/mahasiswa_gui.csv.
    2. Membaca folder pada setiap baris.
    3. Jika folder berisi dataset/recognition/TA/... maka kelas diisi TA.
    4. Jika folder tidak jelas, label dicocokkan dengan class_indices_*.json.
    5. Menghapus duplikasi ringan: jika ada baris sama tetapi satu kosong kelas dan satu sudah benar,
       yang kosong diabaikan.

Tidak mengubah model.
Tidak menghapus dataset.
Tidak mengubah anti-spoofing.

Cara pakai di Raspberry Pi:
    cd <project-root>/src
    python3 fix_roster_class_column_from_folder.py
"""

from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
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


def class_slug(kelas: str) -> str:
    return kelas.strip().lower()


def load_label_to_class(project_root: Path):
    label_to_class = {}

    for kelas in CLASSES:
        path = project_root / "models" / f"class_indices_{class_slug(kelas)}.json"

        if not path.exists():
            continue

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        for label in data.keys():
            label_to_class[str(label).strip()] = kelas

    return label_to_class


def infer_class_from_folder(folder_text: str):
    folder_text = str(folder_text or "").replace("\\", "/").strip()
    parts = [p for p in folder_text.split("/") if p]

    for i, part in enumerate(parts):
        if part == "recognition" and i + 1 < len(parts):
            candidate = parts[i + 1]
            for kelas in CLASSES:
                if candidate.lower() == kelas.lower():
                    return kelas

    # fallback: jika isi folder langsung diawali nama kelas
    for part in parts:
        for kelas in CLASSES:
            if part.lower() == kelas.lower():
                return kelas

    return ""


def infer_label_from_row(row):
    folder = str(row.get("folder", "") or "").replace("\\", "/").strip()
    if folder:
        parts = [p for p in folder.split("/") if p]
        if parts:
            return parts[-1].strip()

    # fallback dari nama/nim
    nim = str(row.get("nim", "") or "").strip()
    nama = str(row.get("nama", "") or "").strip()
    if nim and nama:
        return f"{nim}_{nama.replace(' ', '_')}"
    if nama:
        return nama
    if nim:
        return nim
    return ""


def backup_file(path: Path):
    if not path.exists():
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}_backup_before_fix_class_column_{ts}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def main():
    parser = argparse.ArgumentParser(description="Fix kelas column in mahasiswa_gui.csv.")
    parser.add_argument("--project_root", type=str, default=str(get_project_root()))
    parser.add_argument("--student_csv", type=str, default="")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    student_csv = Path(args.student_csv) if args.student_csv else project_root / "database" / "mahasiswa_gui.csv"

    print("=" * 90)
    print("FIX ROSTER CLASS COLUMN")
    print("=" * 90)
    print(f"Project root : {project_root}")
    print(f"Student CSV  : {student_csv}")
    print("=" * 90)

    if not student_csv.exists():
        print(f"ERROR: mahasiswa_gui.csv tidak ditemukan: {student_csv}")
        sys.exit(1)

    with student_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or ["kelas", "nim", "nama", "folder", "created_at"]
        rows = list(reader)

    required = ["kelas", "nim", "nama", "folder", "created_at"]
    for field in required:
        if field not in fieldnames:
            fieldnames.append(field)

    backup = backup_file(student_csv)
    label_to_class = load_label_to_class(project_root)

    fixed = 0
    still_empty = 0

    normalized_rows = []

    for row in rows:
        for field in fieldnames:
            row.setdefault(field, "")

        old_kelas = str(row.get("kelas", "") or "").strip()

        if old_kelas in CLASSES:
            normalized_rows.append(row)
            continue

        inferred = infer_class_from_folder(row.get("folder", ""))

        if not inferred:
            label = infer_label_from_row(row)
            inferred = label_to_class.get(label, "")

        if inferred:
            row["kelas"] = inferred
            fixed += 1
        else:
            still_empty += 1

        normalized_rows.append(row)

    # Deduplicate: prefer rows with valid kelas.
    seen = set()
    deduped = []
    removed_duplicates = 0

    # Sort so valid kelas rows come first.
    normalized_rows.sort(
        key=lambda r: (
            0 if str(r.get("kelas", "")).strip() in CLASSES else 1,
            str(r.get("kelas", "")),
            str(r.get("nim", "")),
            str(r.get("folder", "")),
        )
    )

    for row in normalized_rows:
        kelas = str(row.get("kelas", "") or "").strip()
        nim = str(row.get("nim", "") or "").strip()
        folder = str(row.get("folder", "") or "").strip()
        nama = str(row.get("nama", "") or "").strip()

        # Key longgar agar baris kosong kelas duplikat bisa dibuang jika sudah ada yang valid.
        key = (kelas if kelas in CLASSES else "", nim, folder or nama)

        # Untuk baris kosong kelas, jangan hilangkan terlalu agresif jika belum ada baris valid.
        if key in seen:
            removed_duplicates += 1
            continue

        seen.add(key)
        deduped.append(row)

    with student_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(deduped)

    print("\nSUMMARY")
    print("=" * 90)
    print(f"Rows before        : {len(rows)}")
    print(f"Class fixed        : {fixed}")
    print(f"Still empty        : {still_empty}")
    print(f"Duplicates removed : {removed_duplicates}")
    print(f"Rows after         : {len(deduped)}")
    print(f"Backup             : {backup}")
    print("=" * 90)
    print("Selesai.")
    print("Tutup GUI, buka ulang, lalu klik Refresh Tabel.")


if __name__ == "__main__":
    main()
