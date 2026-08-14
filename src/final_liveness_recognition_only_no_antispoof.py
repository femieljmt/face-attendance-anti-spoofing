import cv2
import time
import math
import json
import random
import argparse
import warnings
from pathlib import Path

import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

warnings.filterwarnings(
    "ignore",
    message=".*SymbolDatabase.GetPrototype.*",
    category=UserWarning,
)


# ============================================================
# BASIC UTILS
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


def get_clahe_params(level):
    if level == "none":
        return None
    if level == "light":
        return 2.0, (8, 8)
    if level == "medium":
        return 3.0, (8, 8)
    if level == "strong":
        return 4.0, (6, 6)
    return 3.0, (8, 8)


def apply_clahe_bgr(image, level="medium"):
    params = get_clahe_params(level)

    if params is None:
        return image

    clip_limit, tile_grid_size = params

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    l_clahe = clahe.apply(l_channel)
    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def square_crop(image, face_box, scale=1.10, y_shift=0.00):
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


def detect_face_box_mediapipe(frame_bgr, face_detector):
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


def get_face_tilt_angle(face_landmarks, width, height):
    lm = face_landmarks.landmark

    left_eye_x = ((lm[362].x + lm[263].x) / 2.0) * width
    left_eye_y = ((lm[362].y + lm[263].y) / 2.0) * height

    right_eye_x = ((lm[33].x + lm[133].x) / 2.0) * width
    right_eye_y = ((lm[33].y + lm[133].y) / 2.0) * height

    dx = left_eye_x - right_eye_x
    dy = left_eye_y - right_eye_y

    return float(np.degrees(np.arctan2(dy, dx)))


# ============================================================
# RECOGNITION FUNCTIONS
# ============================================================

def load_class_indices(path):
    with open(path, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}


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
# STATE
# ============================================================

def reset_all_state():
    return {
        "system_state": "WAIT_FACE",

        "face_start_time": None,
        "neutral_mouth_values": [],
        "neutral_mouth_norm": None,
        "challenge": None,
        "challenge_start_time": None,
        "success_counter": 0,
        "failed_attempts": 0,
        "liveness_success": False,

        "recognition_done": False,
        "recognition_result": "Belum dikenali",
        "recognition_confidence": 0.0,
        "recognition_buffer": [],
        "top3_text": "Top3: -",
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Final sementara: Liveness + MobileNet recognition TANPA anti-spoofing."
    )

    parser.add_argument("--kelas", type=str, default=None, choices=["PCD", "Pempros", "PSD"])
    parser.add_argument("--camera", type=int, default=0)

    parser.add_argument("--crop_scale", type=float, default=1.10)
    parser.add_argument("--crop_y_shift", type=float, default=0.00)

    parser.add_argument("--recognition_threshold", type=float, default=0.50)
    parser.add_argument("--recognition_frames", type=int, default=20)
    parser.add_argument("--clahe_level", type=str, default="medium", choices=["none", "light", "medium", "strong"])

    parser.add_argument("--max_tilt", type=float, default=20.0)
    parser.add_argument("--min_face_ratio", type=float, default=0.04)
    parser.add_argument("--max_face_ratio", type=float, default=0.45)

    args = parser.parse_args()

    if args.kelas is None:
        print("\nPilih kelas yang akan diabsensi:")
        print("1. PCD")
        print("2. Pempros")
        print("3. PSD")

        pilihan = input("Masukkan pilihan kelas [1/2/3]: ").strip()

        kelas_map = {
            "1": "PCD",
            "2": "Pempros",
            "3": "PSD"
        }

        if pilihan not in kelas_map:
            print("Pilihan tidak valid. Program dihentikan.")
            return

        args.kelas = kelas_map[pilihan]

    project_root = Path(__file__).resolve().parents[1]

    recognition_model_path = project_root / "models" / f"face_recognition_mobilenet_{args.kelas.lower()}.h5"
    class_indices_path = project_root / "models" / f"class_indices_{args.kelas.lower()}.json"

    if not recognition_model_path.exists():
        print("Model face recognition tidak ditemukan:")
        print(recognition_model_path)
        print("\nTraining dulu:")
        print(f"python src/train_recognition_mobilenetv2_strict.py --kelas {args.kelas} --epochs 25 --batch_size 4")
        return

    if not class_indices_path.exists():
        print("Class indices tidak ditemukan:")
        print(class_indices_path)
        return

    print("=" * 90)
    print("FINAL SEMENTARA - LIVENESS + RECOGNITION ONLY")
    print("=" * 90)
    print(f"Kelas dipilih          : {args.kelas}")
    print(f"Recognition model      : {recognition_model_path}")
    print(f"Recognition threshold  : {args.recognition_threshold}")
    print(f"Recognition frames     : {args.recognition_frames}")
    print(f"Anti-spoofing          : NONAKTIF")
    print("=" * 90)

    recognition_model = load_model(str(recognition_model_path))
    index_to_class = load_class_indices(class_indices_path)

    print("Daftar label recognition:")
    for idx, label in index_to_class.items():
        print(f"  {idx}: {label}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Webcam tidak terbaca.")
        return

    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    challenges = ["SMILE", "BLINK_BOTH", "BLINK_LEFT", "BLINK_RIGHT"]
    challenge_instruction = {
        "SMILE": "Silakan SENYUM",
        "BLINK_BOTH": "KEDIPKAN kedua mata",
        "BLINK_LEFT": "KEDIPKAN mata kiri",
        "BLINK_RIGHT": "KEDIPKAN mata kanan"
    }

    required_stable_time = 2.0
    calibration_frame_target = 25
    challenge_timeout = 10.0
    max_failed_attempts = 3
    required_success_frames = 5

    blink_ear_threshold = 0.200
    open_eye_ear_threshold = 0.230
    smile_delta_threshold = 0.030

    state = reset_all_state()

    print("\nProgram berjalan.")
    print("q = keluar | r = reset | c = ganti challenge")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        status_text = "Arahkan wajah ke kamera"
        status_color = (0, 0, 255)

        left_ear_text = "Left EAR: -"
        right_ear_text = "Right EAR: -"
        mouth_norm_text = "MouthNorm: -"
        challenge_text = "Challenge: -"
        debug_text = ""
        crop = None
        crop_box = None
        tilt_angle = 0.0

        face_box = detect_face_box_mediapipe(frame, face_detector)

        if face_box is None:
            state = reset_all_state()
            status_text = "Tidak ada wajah terdeteksi"
            status_color = (0, 0, 255)

        else:
            x, y, w, h = face_box

            face_ratio = (w * h) / float(width * height)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            crop, crop_box = square_crop(
                frame,
                (x, y, w, h),
                scale=args.crop_scale,
                y_shift=args.crop_y_shift
            )

            if crop_box is not None:
                cx1, cy1, cx2, cy2 = crop_box
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)

            if face_ratio < args.min_face_ratio:
                state = reset_all_state()
                status_text = "Wajah terlalu jauh"
                status_color = (0, 0, 255)

            elif face_ratio > args.max_face_ratio:
                state = reset_all_state()
                status_text = "Wajah terlalu dekat"
                status_color = (0, 0, 255)

            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if not results.multi_face_landmarks:
                    state = reset_all_state()
                    status_text = "Landmark wajah belum stabil"
                    status_color = (0, 0, 255)

                else:
                    face_landmarks = results.multi_face_landmarks[0]
                    tilt_angle = get_face_tilt_angle(face_landmarks, width, height)

                    if abs(tilt_angle) > args.max_tilt:
                        status_text = f"Jangan miringkan kepala ({tilt_angle:.1f})"
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

                        left_ear_text = f"Left EAR: {left_ear:.3f}"
                        right_ear_text = f"Right EAR: {right_ear:.3f}"
                        mouth_norm_text = f"MouthNorm: {mouth_norm:.3f}"

                        # ============================================================
                        # STAGE 1: LIVENESS
                        # ============================================================
                        if not state["liveness_success"]:
                            state["system_state"] = "LIVENESS"

                            if state["face_start_time"] is None:
                                state["face_start_time"] = time.time()

                            elapsed_face_time = time.time() - state["face_start_time"]

                            if elapsed_face_time < required_stable_time:
                                status_text = f"Tahan wajah stabil: {elapsed_face_time:.1f}/{required_stable_time:.1f} detik"
                                status_color = (0, 255, 255)

                                state["neutral_mouth_values"] = []
                                state["neutral_mouth_norm"] = None
                                state["challenge"] = None
                                state["challenge_start_time"] = None
                                state["success_counter"] = 0

                            elif state["neutral_mouth_norm"] is None:
                                state["neutral_mouth_values"].append(mouth_norm)

                                status_text = (
                                    f"Kalibrasi wajah netral: "
                                    f"{len(state['neutral_mouth_values'])}/{calibration_frame_target}"
                                )
                                status_color = (255, 255, 0)

                                if len(state["neutral_mouth_values"]) >= calibration_frame_target:
                                    state["neutral_mouth_norm"] = float(np.median(state["neutral_mouth_values"]))
                                    state["neutral_mouth_values"] = []
                                    state["challenge"] = random.choice(challenges)
                                    state["challenge_start_time"] = time.time()
                                    state["success_counter"] = 0

                            else:
                                if state["challenge"] is None:
                                    state["challenge"] = random.choice(challenges)
                                    state["challenge_start_time"] = time.time()
                                    state["success_counter"] = 0

                                challenge = state["challenge"]
                                challenge_elapsed = time.time() - state["challenge_start_time"]
                                remaining_time = max(0, challenge_timeout - challenge_elapsed)

                                status_text = f"{challenge_instruction[challenge]} ({remaining_time:.1f} detik)"
                                status_color = (255, 255, 0)
                                challenge_text = f"Challenge: {challenge}"

                                smile_delta = mouth_norm - state["neutral_mouth_norm"]

                                is_smile = smile_delta > smile_delta_threshold
                                is_blink_both = (
                                    left_ear < blink_ear_threshold and
                                    right_ear < blink_ear_threshold
                                )
                                is_blink_left = (
                                    left_ear < blink_ear_threshold and
                                    right_ear > open_eye_ear_threshold
                                )
                                is_blink_right = (
                                    right_ear < blink_ear_threshold and
                                    left_ear > open_eye_ear_threshold
                                )

                                challenge_success = False

                                if challenge == "SMILE":
                                    challenge_success = is_smile
                                    debug_text = f"SmileDelta: {smile_delta:.3f}"
                                elif challenge == "BLINK_BOTH":
                                    challenge_success = is_blink_both
                                    debug_text = "Target: kedua mata tertutup"
                                elif challenge == "BLINK_LEFT":
                                    challenge_success = is_blink_left
                                    debug_text = "Target: mata kiri tertutup"
                                elif challenge == "BLINK_RIGHT":
                                    challenge_success = is_blink_right
                                    debug_text = "Target: mata kanan tertutup"

                                if challenge_success:
                                    state["success_counter"] += 1
                                    status_text = f"Challenge terdeteksi: {state['success_counter']}/{required_success_frames}"
                                    status_color = (0, 255, 0)
                                else:
                                    state["success_counter"] = max(0, state["success_counter"] - 1)

                                if state["success_counter"] >= required_success_frames:
                                    state["liveness_success"] = True
                                    state["system_state"] = "RECOGNITION"
                                    state["recognition_buffer"] = []
                                    status_text = "Liveness berhasil - lanjut recognition"
                                    status_color = (0, 255, 0)

                                if challenge_elapsed > challenge_timeout and not state["liveness_success"]:
                                    state["failed_attempts"] += 1

                                    if state["failed_attempts"] >= max_failed_attempts:
                                        status_text = "Liveness gagal - proses ditolak"
                                        status_color = (0, 0, 255)
                                        cv2.putText(frame, status_text, (20, 40),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                                        cv2.imshow("Liveness + Recognition Only", frame)
                                        cv2.waitKey(1000)

                                        state = reset_all_state()
                                    else:
                                        state["challenge"] = random.choice(challenges)
                                        state["challenge_start_time"] = time.time()
                                        state["success_counter"] = 0
                                        status_text = f"Timeout - ulangi challenge ({state['failed_attempts']}/{max_failed_attempts})"
                                        status_color = (0, 0, 255)

                        # ============================================================
                        # STAGE 2: RECOGNITION ONLY
                        # ============================================================
                        elif not state["recognition_done"]:
                            state["system_state"] = "RECOGNITION"

                            if crop is not None and crop.size > 0:
                                probs = predict_recognition_probs(
                                    model=recognition_model,
                                    face_crop=crop,
                                    img_size=224,
                                    clahe_level=args.clahe_level
                                )

                                state["recognition_buffer"].append(probs)

                                avg_probs = np.mean(np.array(state["recognition_buffer"]), axis=0)
                                state["top3_text"] = "Top3: " + format_top3(avg_probs, index_to_class)

                                status_text = f"Recognition frame: {len(state['recognition_buffer'])}/{args.recognition_frames}"
                                status_color = (255, 255, 0)

                                if len(state["recognition_buffer"]) >= args.recognition_frames:
                                    best_idx = int(np.argmax(avg_probs))
                                    best_conf = float(avg_probs[best_idx])
                                    best_label = index_to_class.get(best_idx, "UNKNOWN")

                                    state["recognition_confidence"] = best_conf

                                    if best_conf >= args.recognition_threshold:
                                        state["recognition_result"] = best_label
                                    else:
                                        state["recognition_result"] = "Tidak dikenal"

                                    state["recognition_done"] = True

                        else:
                            state["system_state"] = "DONE"

                            if state["recognition_result"] == "Tidak dikenal":
                                status_text = f"Tidak dikenal ({state['recognition_confidence']:.2f})"
                                status_color = (0, 0, 255)
                            else:
                                status_text = f"Absensi valid: {state['recognition_result']} ({state['recognition_confidence']:.2f})"
                                status_color = (0, 255, 0)

        # ============================================================
        # DISPLAY
        # ============================================================
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.70, status_color, 2)
        cv2.putText(frame, f"State: {state['system_state']}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, f"Kelas: {args.kelas}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, "Anti-spoofing: OFF", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"Failed Liveness: {state['failed_attempts']}/{max_failed_attempts}", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
        cv2.putText(frame, left_ear_text, (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        cv2.putText(frame, right_ear_text, (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        cv2.putText(frame, mouth_norm_text, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        cv2.putText(frame, challenge_text, (20, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        cv2.putText(frame, f"Tilt:{tilt_angle:.1f} | crop_scale:{args.crop_scale:.2f}", (20, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

        if state["top3_text"]:
            cv2.putText(frame, state["top3_text"], (20, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 2)

        if debug_text:
            cv2.putText(frame, debug_text, (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 2)

        cv2.putText(frame, "q: keluar | r: reset | c: ganti challenge",
                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)

        cv2.imshow("Liveness + Recognition Only", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            state = reset_all_state()
            print("Reset.")

        elif key == ord("c"):
            if state["neutral_mouth_norm"] is not None and not state["liveness_success"]:
                state["challenge"] = random.choice(challenges)
                state["challenge_start_time"] = time.time()
                state["success_counter"] = 0
                print("Challenge diganti:", state["challenge"])

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    face_detector.close()


if __name__ == "__main__":
    main()
