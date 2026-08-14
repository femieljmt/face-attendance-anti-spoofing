import cv2
import time
import math
import json
import csv
import random
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*", category=UserWarning)


# ============================================================
# BASIC UTILITIES
# ============================================================

def distance_2d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_point(face_landmarks, index, width, height):
    lm = face_landmarks.landmark[index]
    return int(lm.x * width), int(lm.y * height)


def calculate_ear(face_landmarks, eye_indices, width, height):
    p1 = get_point(face_landmarks, eye_indices[0], width, height)
    p2 = get_point(face_landmarks, eye_indices[1], width, height)
    p3 = get_point(face_landmarks, eye_indices[2], width, height)
    p4 = get_point(face_landmarks, eye_indices[3], width, height)
    p5 = get_point(face_landmarks, eye_indices[4], width, height)
    p6 = get_point(face_landmarks, eye_indices[5], width, height)

    vertical_1 = distance_2d(p2, p6)
    vertical_2 = distance_2d(p3, p5)
    horizontal = distance_2d(p1, p4)

    return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)


def get_face_tilt_angle(face_landmarks, width, height):
    lm = face_landmarks.landmark

    left_eye_x = ((lm[362].x + lm[263].x) / 2.0) * width
    left_eye_y = ((lm[362].y + lm[263].y) / 2.0) * height

    right_eye_x = ((lm[33].x + lm[133].x) / 2.0) * width
    right_eye_y = ((lm[33].y + lm[133].y) / 2.0) * height

    dx = left_eye_x - right_eye_x
    dy = left_eye_y - right_eye_y

    return float(np.degrees(np.arctan2(dy, dx)))


def apply_clahe_bgr(image, level="medium"):
    if level == "none":
        return image

    if level == "light":
        clip_limit, tile_grid_size = 2.0, (8, 8)
    elif level == "strong":
        clip_limit, tile_grid_size = 4.0, (6, 6)
    else:
        clip_limit, tile_grid_size = 3.0, (8, 8)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)
    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def square_crop(image, face_box, scale=1.20, y_shift=0.00):
    x, y, w, h = face_box
    img_h, img_w = image.shape[:2]

    cx = x + w / 2
    cy = y + h / 2 + y_shift * h
    side = int(max(w, h) * scale)

    x1 = int(cx - side / 2)
    y1 = int(cy - side / 2)
    x2 = x1 + side
    y2 = y1 + side

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = img_w
    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = img_h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    crop = image[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)



def tight_facemesh_crop(image_bgr, face_mesh, pad_x=0.06, pad_top=0.04, pad_bottom=0.01):
    """
    Crop khusus recognition kelas TA.

    Harus sama dengan preprocessing/training TA:
        - tight FaceMesh crop
        - pad_x      = 0.06
        - pad_top    = 0.04
        - pad_bottom = 0.01
        - CLAHE      = light pada predict_recognition_probs()
        - resize     = 224x224
    """
    if face_mesh is None:
        return None, None

    img_h, img_w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0].landmark

    xs = []
    ys = []

    for lm in landmarks:
        x = int(lm.x * img_w)
        y = int(lm.y * img_h)

        if 0 <= x < img_w and 0 <= y < img_h:
            xs.append(x)
            ys.append(y)

    if not xs or not ys:
        return None, None

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    face_w = x_max - x_min
    face_h = y_max - y_min

    if face_w <= 0 or face_h <= 0:
        return None, None

    x1 = int(x_min - pad_x * face_w)
    x2 = int(x_max + pad_x * face_w)
    y1 = int(y_min - pad_top * face_h)
    y2 = int(y_max + pad_bottom * face_h)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = image_bgr[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        return None, None

    return crop, (x1, y1, x2, y2)


# ============================================================
# FACE DETECTION
# ============================================================

def detect_face_haar(frame_bgr, haar_detector, min_face_size=70):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    faces = haar_detector.detectMultiScale(
        gray,
        scaleFactor=1.10,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size)
    )

    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
    x, y, w, h = faces[0]
    return int(x), int(y), int(w), int(h)


def detect_face_mediapipe(frame_bgr, face_detector):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)

    if not results.detections:
        return None

    boxes = []

    for det in results.detections:
        bbox = det.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        x = max(0, x)
        y = max(0, y)
        bw = min(w - x, bw)
        bh = min(h - y, bh)

        if bw > 0 and bh > 0:
            boxes.append((x, y, bw, bh))

    if not boxes:
        return None

    return max(boxes, key=lambda b: b[2] * b[3])


def is_face_stable(prev_box, current_box, movement_threshold=0.12):
    if prev_box is None or current_box is None:
        return False

    px, py, pw, ph = prev_box
    cx, cy, cw, ch = current_box

    prev_center = np.array([px + pw / 2, py + ph / 2])
    curr_center = np.array([cx + cw / 2, cy + ch / 2])

    center_dist = np.linalg.norm(curr_center - prev_center)
    size_ref = max(pw, ph, cw, ch, 1)

    area_prev = pw * ph
    area_curr = cw * ch
    area_change = abs(area_curr - area_prev) / max(area_prev, 1)

    return (center_dist / size_ref) < movement_threshold and area_change < 0.35


# ============================================================
# MODEL FUNCTIONS
# ============================================================

def load_class_indices(path):
    with open(path, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    return {int(v): str(k) for k, v in class_indices.items()}


def load_spoof_threshold(path, default=0.10):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("threshold", default))
    except Exception:
        return default


def predict_spoof_probability(model, face_crop, img_size=224):
    face = cv2.resize(face_crop, (img_size, img_size))
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    arr = face_rgb.astype("float32")
    arr = np.expand_dims(arr, axis=0)
    # Model anti-spoofing V3 sudah punya Rescaling layer di dalam model.
    return float(model.predict(arr, verbose=0)[0][0])


def predict_recognition_probs(model, face_crop, img_size=224, clahe_level="medium"):
    face = cv2.resize(face_crop, (img_size, img_size))
    face = apply_clahe_bgr(face, level=clahe_level)

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    arr = face_rgb.astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    return model.predict(arr, verbose=0)[0]


def format_top3(avg_probs, index_to_class):
    top_indices = np.argsort(avg_probs)[::-1][:3]
    parts = []
    for idx in top_indices:
        label = index_to_class.get(int(idx), "UNKNOWN")
        parts.append(f"{label[:16]}:{avg_probs[idx]:.2f}")
    return " | ".join(parts)


# ============================================================
# ATTENDANCE LOGGING
# ============================================================

def save_attendance_log(csv_path, kelas, result, recognition_conf, spoof_prob, status):
    """Save attendance log safely.

    If the target CSV is locked by Excel/another process, write to a fallback file
    so the backend does not crash during a running attendance session.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        kelas,
        result,
        f"{recognition_conf:.4f}",
        f"{spoof_prob:.4f}",
        status,
    ]

    header = ["timestamp", "kelas", "result", "recognition_confidence", "spoof_probability", "status"]

    targets = [csv_path]

    # Fallback if the selected log file is locked/permission denied.
    fallback = csv_path.parent / f"attendance_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    targets.append(fallback)

    last_error = None

    for target in targets:
        try:
            file_exists = target.exists()

            with open(target, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(header)
                writer.writerow(row)

            return target

        except PermissionError as e:
            last_error = e
            print(f"WARNING: log file terkunci/tidak bisa ditulis: {target}")
            continue

    print(f"ERROR: gagal menyimpan log absensi: {last_error}")
    return None


# ============================================================
# STATE
# ============================================================

def make_initial_state():
    return {
        "state": "IDLE",
        "last_face_box": None,
        "stable_start_time": None,
        "no_face_start_time": None,
        "neutral_mouth_values": [],
        "neutral_mouth_norm": None,
        "challenge": None,
        "challenge_start_time": None,
        "challenge_success_counter": 0,
        "liveness_failed_attempts": 0,
        "liveness_pass": False,
        "spoof_probs": [],
        "spoof_probability": 0.0,
        "anti_spoof_result": "-",
        "recognition_probs": [],
        "recognition_result": "-",
        "recognition_confidence": 0.0,
        "top3_text": "Top3: -",
        "result_saved": False,
        "final_status": "-",
        "message": "Arahkan wajah ke kamera",
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="FINAL lightweight attendance: Haar trigger + Liveness + Anti-spoofing + Recognition.")
    parser.add_argument("--kelas", type=str, default=None, choices=["PCD", "TA", "Pempros", "PSD"])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--no_flip", action="store_true")
    parser.add_argument("--stable_seconds", type=float, default=1.2)
    parser.add_argument("--face_leave_seconds", type=float, default=1.5)
    parser.add_argument("--min_face_ratio", type=float, default=0.035)
    parser.add_argument("--max_face_ratio", type=float, default=0.50)
    parser.add_argument("--crop_scale", type=float, default=1.20)
    parser.add_argument("--crop_y_shift", type=float, default=0.00)
    parser.add_argument("--mp_det_conf", type=float, default=0.35)
    parser.add_argument("--max_tilt", type=float, default=22.0)
    parser.add_argument("--challenge_set", type=str, default="simple", choices=["simple", "all"])
    parser.add_argument("--challenge_timeout", type=float, default=10.0)
    parser.add_argument("--required_liveness_frames", type=int, default=4)
    parser.add_argument("--calibration_frames", type=int, default=18)
    parser.add_argument("--anti_spoof_threshold", type=float, default=None)
    parser.add_argument("--anti_spoof_frames", type=int, default=5)
    parser.add_argument("--recognition_threshold", type=float, default=0.35)
    parser.add_argument("--recognition_frames", type=int, default=7)
    parser.add_argument("--clahe_level", type=str, default="medium", choices=["none", "light", "medium", "strong"])
    parser.add_argument("--save_log", action="store_true")
    parser.add_argument("--log_path", type=str, default="database/absensi_runtime.csv")
    parser.add_argument("--headless", action="store_true")

    # Optional testing modes from GUI/CLI.
    # --disable_liveness   : skip FaceMesh challenge, useful for recognition-only or speed testing.
    # --disable_antispoof  : skip anti-spoofing MobileNet, useful for recognition-only or liveness-only testing.
    # --recognition_only   : shortcut for disabling both liveness and anti-spoofing.
    parser.add_argument("--disable_liveness", action="store_true")
    parser.add_argument("--disable_antispoof", action="store_true")
    parser.add_argument("--recognition_only", action="store_true")

    args = parser.parse_args()

    if args.recognition_only:
        args.disable_liveness = True
        args.disable_antispoof = True

    if args.kelas is None:
        print("\nPilih kelas:")
        print("1. PCD")
        print("2. TA")
        print("3. Pempros")
        print("4. PSD")
        pilihan = input("Masukkan pilihan [1/2/3/4]: ").strip()
        kelas_map = {"1": "PCD", "2": "TA", "3": "Pempros", "4": "PSD"}
        if pilihan not in kelas_map:
            print("Pilihan tidak valid.")
            return
        args.kelas = kelas_map[pilihan]

    # Preset khusus kelas TA.
    # Tidak mengubah anti-spoofing. Hanya menyesuaikan recognition dengan model TA.
    if args.kelas == "TA":
        if abs(args.recognition_threshold - 0.35) < 1e-9:
            args.recognition_threshold = 0.25
        if args.recognition_frames == 7:
            args.recognition_frames = 10
        if args.clahe_level == "medium":
            args.clahe_level = "light"

    project_root = Path(__file__).resolve().parents[1]

    anti_spoof_model_path = project_root / "models" / "anti_spoofing_mobilenet_real_spoof.h5"
    anti_spoof_threshold_path = project_root / "models" / "anti_spoofing_threshold.json"
    recognition_model_path = project_root / "models" / f"face_recognition_mobilenet_{args.kelas.lower()}.h5"
    class_indices_path = project_root / "models" / f"class_indices_{args.kelas.lower()}.json"

    required_files = [recognition_model_path, class_indices_path]
    if not args.disable_antispoof:
        required_files.append(anti_spoof_model_path)

    for needed in required_files:
        if not needed.exists():
            print("File tidak ditemukan:", needed)
            return

    spoof_threshold = load_spoof_threshold(anti_spoof_threshold_path, default=0.10)
    if args.anti_spoof_threshold is not None:
        spoof_threshold = args.anti_spoof_threshold

    print("=" * 90)
    print("FINAL ATTENDANCE LIGHTWEIGHT - RASPBERRY PI READY")
    print("=" * 90)
    print(f"Kelas                : {args.kelas}")
    print(f"Camera               : {args.camera}")
    print(f"Resolution/FPS        : {args.width}x{args.height} @ {args.fps}")
    print(f"Liveness active       : {not args.disable_liveness}")
    print(f"Anti-spoof active     : {not args.disable_antispoof}")
    print(f"Anti-spoof threshold  : {spoof_threshold}")
    print(f"Recognition threshold : {args.recognition_threshold}")
    print(f"Recognition frames    : {args.recognition_frames}")
    print(f"CLAHE level           : {args.clahe_level}")
    print(f"Challenge set         : {args.challenge_set}")
    if args.kelas == "TA":
        print("TA recognition crop   : tight FaceMesh, pad_x=0.06, pad_top=0.04, pad_bottom=0.01")
    print("=" * 90)

    print("\nLoading models...")
    anti_spoof_model = None
    if not args.disable_antispoof:
        anti_spoof_model = load_model(str(anti_spoof_model_path))

    recognition_model = load_model(str(recognition_model_path))
    index_to_class = load_class_indices(class_indices_path)

    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    haar_detector = cv2.CascadeClassifier(haar_path)
    if haar_detector.empty():
        print("Haar Cascade gagal dimuat:", haar_path)
        return

    mp_face_detection = mp.solutions.face_detection
    mp_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=args.mp_det_conf)

    face_mesh = None
    if (not args.disable_liveness) or args.kelas == "TA":
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    challenges = ["SMILE", "BLINK_BOTH"] if args.challenge_set == "simple" else ["SMILE", "BLINK_BOTH", "BLINK_LEFT", "BLINK_RIGHT"]
    challenge_instruction = {
        "SMILE": "Silakan SENYUM",
        "BLINK_BOTH": "KEDIPKAN kedua mata",
        "BLINK_LEFT": "KEDIPKAN mata kiri",
        "BLINK_RIGHT": "KEDIPKAN mata kanan",
    }

    blink_ear_threshold = 0.200
    open_eye_ear_threshold = 0.230
    smile_delta_threshold = 0.030

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Kamera tidak terbuka.")
        return

    state = make_initial_state()
    log_path = project_root / args.log_path

    print("\nProgram berjalan.")
    print("q = keluar | r = reset")
    print("Haar aktif sebagai trigger ringan.")
    if args.disable_liveness:
        print("MODE: Liveness NONAKTIF.")
    else:
        print("FaceMesh hanya aktif saat LIVENESS_CHECK.")

    if args.disable_antispoof:
        print("MODE: Anti-spoofing NONAKTIF.")

    if args.recognition_only:
        print("MODE: RECOGNITION ONLY.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame tidak terbaca.")
            break

        if not args.no_flip:
            frame = cv2.flip(frame, 1)

        display = frame.copy()
        height, width = frame.shape[:2]
        status_color = (0, 255, 255)
        detail_text = ""
        crop_box = None

        haar_box = detect_face_haar(frame, haar_detector)
        if haar_box is not None:
            hx, hy, hw, hh = haar_box
            cv2.rectangle(display, (hx, hy), (hx + hw, hy + hh), (120, 255, 120), 2)
            face_ratio = (hw * hh) / float(width * height)
        else:
            face_ratio = 0.0

        # Reset saat wajah hilang
        if haar_box is None:
            if state["no_face_start_time"] is None:
                state["no_face_start_time"] = time.time()
            no_face_duration = time.time() - state["no_face_start_time"]
            if state["state"] == "WAIT_FACE_LEAVE" and no_face_duration >= args.face_leave_seconds:
                state = make_initial_state()
            elif state["state"] not in ["IDLE", "WAIT_FACE_LEAVE"] and no_face_duration >= args.face_leave_seconds:
                state = make_initial_state()
        else:
            state["no_face_start_time"] = None

        # ========================================================
        # STATE MACHINE
        # ========================================================
        if state["state"] == "IDLE":
            state["message"] = "Arahkan wajah ke kamera"
            status_color = (0, 255, 255)

            if haar_box is not None:
                if face_ratio < args.min_face_ratio:
                    state["message"] = "Wajah terlalu jauh"
                    status_color = (0, 0, 255)
                    state["stable_start_time"] = None
                elif face_ratio > args.max_face_ratio:
                    state["message"] = "Wajah terlalu dekat"
                    status_color = (0, 0, 255)
                    state["stable_start_time"] = None
                else:
                    if state["last_face_box"] is None:
                        state["last_face_box"] = haar_box
                        state["stable_start_time"] = time.time()
                        state["message"] = "Tahan wajah stabil"
                    elif is_face_stable(state["last_face_box"], haar_box):
                        if state["stable_start_time"] is None:
                            state["stable_start_time"] = time.time()
                        stable_duration = time.time() - state["stable_start_time"]
                        state["message"] = f"Tahan wajah stabil: {stable_duration:.1f}/{args.stable_seconds:.1f} detik"
                        status_color = (255, 255, 0)
                        if stable_duration >= args.stable_seconds:
                            if args.disable_liveness:
                                state["liveness_pass"] = True
                                state["message"] = "Liveness dilewati - lanjut tahap berikutnya"

                                if args.disable_antispoof:
                                    state["anti_spoof_result"] = "SKIPPED"
                                    state["state"] = "RECOGNITION"
                                    state["recognition_probs"] = []
                                else:
                                    state["state"] = "ANTI_SPOOF_CHECK"
                                    state["spoof_probs"] = []
                            else:
                                state["state"] = "LIVENESS_CHECK"
                                state["neutral_mouth_values"] = []
                                state["neutral_mouth_norm"] = None
                                state["challenge"] = None
                                state["challenge_success_counter"] = 0
                    else:
                        state["stable_start_time"] = time.time()
                        state["last_face_box"] = haar_box
                        state["message"] = "Wajah belum stabil"

        elif state["state"] == "LIVENESS_CHECK":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                state["message"] = "Landmark wajah belum stabil"
                status_color = (0, 0, 255)
            else:
                face_landmarks = results.multi_face_landmarks[0]
                tilt_angle = get_face_tilt_angle(face_landmarks, width, height)

                if abs(tilt_angle) > args.max_tilt:
                    state["message"] = f"Jangan miringkan kepala ({tilt_angle:.1f})"
                    status_color = (0, 255, 255)
                else:
                    left_ear = calculate_ear(face_landmarks, LEFT_EYE, width, height)
                    right_ear = calculate_ear(face_landmarks, RIGHT_EYE, width, height)

                    p_mouth_left = get_point(face_landmarks, 61, width, height)
                    p_mouth_right = get_point(face_landmarks, 291, width, height)
                    p_face_left = get_point(face_landmarks, 234, width, height)
                    p_face_right = get_point(face_landmarks, 454, width, height)

                    mouth_width = distance_2d(p_mouth_left, p_mouth_right)
                    face_width = distance_2d(p_face_left, p_face_right)
                    mouth_norm = mouth_width / (face_width + 1e-6)

                    detail_text = f"EAR L:{left_ear:.3f} R:{right_ear:.3f} | Mouth:{mouth_norm:.3f}"

                    if state["neutral_mouth_norm"] is None:
                        state["neutral_mouth_values"].append(mouth_norm)
                        state["message"] = f"Kalibrasi wajah netral: {len(state['neutral_mouth_values'])}/{args.calibration_frames}"
                        status_color = (255, 255, 0)
                        if len(state["neutral_mouth_values"]) >= args.calibration_frames:
                            state["neutral_mouth_norm"] = float(np.median(state["neutral_mouth_values"]))
                            state["challenge"] = random.choice(challenges)
                            state["challenge_start_time"] = time.time()
                            state["challenge_success_counter"] = 0
                    else:
                        if state["challenge"] is None:
                            state["challenge"] = random.choice(challenges)
                            state["challenge_start_time"] = time.time()
                            state["challenge_success_counter"] = 0

                        challenge = state["challenge"]
                        elapsed = time.time() - state["challenge_start_time"]
                        remaining = max(0, args.challenge_timeout - elapsed)
                        smile_delta = mouth_norm - state["neutral_mouth_norm"]

                        is_smile = smile_delta > smile_delta_threshold
                        is_blink_both = left_ear < blink_ear_threshold and right_ear < blink_ear_threshold
                        is_blink_left = left_ear < blink_ear_threshold and right_ear > open_eye_ear_threshold
                        is_blink_right = right_ear < blink_ear_threshold and left_ear > open_eye_ear_threshold

                        if challenge == "SMILE":
                            challenge_success = is_smile
                            detail_text += f" | SmileDelta:{smile_delta:.3f}"
                        elif challenge == "BLINK_BOTH":
                            challenge_success = is_blink_both
                        elif challenge == "BLINK_LEFT":
                            challenge_success = is_blink_left
                        else:
                            challenge_success = is_blink_right

                        if challenge_success:
                            state["challenge_success_counter"] += 1
                        else:
                            state["challenge_success_counter"] = max(0, state["challenge_success_counter"] - 1)

                        state["message"] = f"{challenge_instruction[challenge]} | {state['challenge_success_counter']}/{args.required_liveness_frames} | {remaining:.1f}s"
                        status_color = (255, 255, 0)

                        if state["challenge_success_counter"] >= args.required_liveness_frames:
                            state["liveness_pass"] = True

                            if args.disable_antispoof:
                                state["anti_spoof_result"] = "SKIPPED"
                                state["state"] = "RECOGNITION"
                                state["recognition_probs"] = []
                                state["message"] = "Liveness berhasil - anti-spoofing dilewati - lanjut recognition"
                            else:
                                state["state"] = "ANTI_SPOOF_CHECK"
                                state["spoof_probs"] = []
                                state["message"] = "Liveness berhasil - cek anti-spoofing"

                            status_color = (0, 255, 0)
                        elif elapsed >= args.challenge_timeout:
                            state["liveness_failed_attempts"] += 1
                            if state["liveness_failed_attempts"] >= 3:
                                state["state"] = "ATTENDANCE_RESULT"
                                state["final_status"] = "DITOLAK_LIVENESS"
                                state["recognition_result"] = "Liveness gagal"
                                state["message"] = "Liveness gagal - absensi ditolak"
                                status_color = (0, 0, 255)
                            else:
                                state["challenge"] = random.choice(challenges)
                                state["challenge_start_time"] = time.time()
                                state["challenge_success_counter"] = 0
                                state["message"] = "Challenge timeout - ulangi"

        elif state["state"] == "ANTI_SPOOF_CHECK":
            if args.disable_antispoof:
                state["anti_spoof_result"] = "SKIPPED"
                state["state"] = "RECOGNITION"
                state["recognition_probs"] = []
                state["message"] = "Anti-spoofing dilewati - lanjut recognition"
                status_color = (0, 255, 0)
                continue

            mp_box = detect_face_mediapipe(frame, mp_detector)
            if mp_box is None:
                state["message"] = "Wajah tidak stabil saat anti-spoofing"
                status_color = (0, 0, 255)
            else:
                crop, crop_box = square_crop(frame, mp_box, scale=args.crop_scale, y_shift=args.crop_y_shift)
                if crop is not None and crop.size > 0:
                    prob = predict_spoof_probability(anti_spoof_model, crop)
                    state["spoof_probs"].append(prob)
                    avg_spoof = float(np.mean(state["spoof_probs"]))
                    state["spoof_probability"] = avg_spoof
                    state["message"] = f"Anti-spoofing: {len(state['spoof_probs'])}/{args.anti_spoof_frames} | spoof_prob={avg_spoof:.3f}"
                    status_color = (255, 255, 0)

                    if len(state["spoof_probs"]) >= args.anti_spoof_frames:
                        if avg_spoof >= spoof_threshold:
                            state["anti_spoof_result"] = "SPOOF"
                            state["state"] = "ATTENDANCE_RESULT"
                            state["final_status"] = "DITOLAK_SPOOF"
                            state["recognition_result"] = "Terdeteksi spoof"
                            state["message"] = f"SPOOF terdeteksi ({avg_spoof:.3f}) - absensi ditolak"
                            status_color = (0, 0, 255)
                        else:
                            state["anti_spoof_result"] = "REAL"
                            state["state"] = "RECOGNITION"
                            state["recognition_probs"] = []
                            state["message"] = f"REAL ({avg_spoof:.3f}) - lanjut recognition"
                            status_color = (0, 255, 0)

        elif state["state"] == "RECOGNITION":
            if args.kelas == "TA":
                crop, crop_box = tight_facemesh_crop(
                    frame,
                    face_mesh,
                    pad_x=0.06,
                    pad_top=0.04,
                    pad_bottom=0.01
                )
                if crop is None:
                    state["message"] = "FaceMesh crop TA gagal saat recognition"
                    status_color = (0, 0, 255)
                    continue
            else:
                mp_box = detect_face_mediapipe(frame, mp_detector)
                if mp_box is None:
                    state["message"] = "Wajah tidak stabil saat recognition"
                    status_color = (0, 0, 255)
                    continue

                crop, crop_box = square_crop(frame, mp_box, scale=args.crop_scale, y_shift=args.crop_y_shift)

            if crop is not None and crop.size > 0:
                probs = predict_recognition_probs(recognition_model, crop, img_size=224, clahe_level=args.clahe_level)
                state["recognition_probs"].append(probs)
                avg_probs = np.mean(np.array(state["recognition_probs"]), axis=0)
                best_idx = int(np.argmax(avg_probs))
                best_conf = float(avg_probs[best_idx])
                best_label = index_to_class.get(best_idx, "UNKNOWN")
                state["top3_text"] = "Top3: " + format_top3(avg_probs, index_to_class)
                state["recognition_confidence"] = best_conf
                state["message"] = f"Recognition: {len(state['recognition_probs'])}/{args.recognition_frames} | {best_label[:18]} ({best_conf:.3f})"
                status_color = (255, 255, 0)

                if len(state["recognition_probs"]) >= args.recognition_frames:
                    if best_conf >= args.recognition_threshold:
                        state["recognition_result"] = best_label
                        state["final_status"] = "ABSENSI_VALID"
                        state["message"] = f"Absensi valid: {best_label} ({best_conf:.3f})"
                        status_color = (0, 255, 0)
                    else:
                        state["recognition_result"] = "Tidak dikenal"
                        state["final_status"] = "DITOLAK_TIDAK_DIKENAL"
                        state["message"] = f"Tidak dikenal ({best_conf:.3f})"
                        status_color = (0, 0, 255)
                    state["state"] = "ATTENDANCE_RESULT"

        elif state["state"] == "ATTENDANCE_RESULT":
            if not state["result_saved"]:
                if args.save_log:
                    save_attendance_log(
                        csv_path=log_path,
                        kelas=args.kelas,
                        result=state["recognition_result"],
                        recognition_conf=state["recognition_confidence"],
                        spoof_prob=state["spoof_probability"],
                        status=state["final_status"],
                    )
                state["result_saved"] = True
            status_color = (0, 255, 0) if state["final_status"] == "ABSENSI_VALID" else (0, 0, 255)
            state["state"] = "WAIT_FACE_LEAVE"

        elif state["state"] == "WAIT_FACE_LEAVE":
            status_color = (0, 255, 0) if state["final_status"] == "ABSENSI_VALID" else (0, 0, 255)
            state["message"] = f"{state['message']} | Silakan keluar dari frame"

        # ========================================================
        # DISPLAY
        # ========================================================
        if crop_box is not None:
            x1, y1, x2, y2 = crop_box
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.putText(display, state["message"], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.putText(display, f"State: {state['state']} | Kelas: {args.kelas}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        live_display = "OFF" if args.disable_liveness else ("PASS" if state["liveness_pass"] else "-")
        spoof_display = "OFF" if args.disable_antispoof else state["anti_spoof_result"]
        cv2.putText(display, f"Liveness: {live_display} | Anti-spoof: {spoof_display} | threshold:{spoof_threshold:.2f}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        cv2.putText(display, f"SpoofProb:{state['spoof_probability']:.3f} | RecConf:{state['recognition_confidence']:.3f} | RecThr:{args.recognition_threshold:.2f}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

        if state["top3_text"]:
            cv2.putText(display, state["top3_text"], (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 2)
        if detail_text:
            cv2.putText(display, detail_text, (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 2)

        cv2.putText(display, "q: keluar | r: reset", (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        if not args.headless:
            cv2.imshow("Final Attendance Lightweight", display)

        key = cv2.waitKey(1) & 0xFF if not args.headless else 255
        if key == ord("q"):
            break
        elif key == ord("r"):
            state = make_initial_state()
            print("Reset manual.")

    cap.release()
    cv2.destroyAllWindows()
    if face_mesh is not None:
        face_mesh.close()
    mp_detector.close()


if __name__ == "__main__":
    main()
