import cv2
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*", category=UserWarning)


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

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)

    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


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


def load_class_indices(path):
    with open(path, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    return {int(v): str(k) for k, v in class_indices.items()}


def predict_recognition_probs(model, face_crop, clahe_level="medium"):
    face = cv2.resize(face_crop, (224, 224))
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
        parts.append(f"{label[:18]}:{avg_probs[idx]:.2f}")

    return " | ".join(parts)


def choose_class_interactive():
    print("\nPilih kelas yang akan dites:")
    print("1. PCD")
    print("2. Pempros")
    print("3. PSD")

    pilihan = input("Masukkan pilihan kelas [1/2/3]: ").strip()

    kelas_map = {
        "1": "PCD",
        "2": "Pempros",
        "3": "PSD"
    }

    return kelas_map.get(pilihan)


def main():
    parser = argparse.ArgumentParser(
        description="Test MobileNet recognition only. Tanpa liveness dan tanpa anti-spoofing."
    )

    parser.add_argument("--kelas", type=str, default=None, choices=["PCD", "Pempros", "PSD"])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--recognition_threshold", type=float, default=0.35)
    parser.add_argument("--recognition_frames", type=int, default=25)
    parser.add_argument("--crop_scale", type=float, default=1.10)
    parser.add_argument("--crop_y_shift", type=float, default=0.00)
    parser.add_argument("--clahe_level", type=str, default="medium", choices=["none", "light", "medium", "strong"])
    parser.add_argument("--min_face_ratio", type=float, default=0.04)
    parser.add_argument("--max_face_ratio", type=float, default=0.45)

    args = parser.parse_args()

    if args.kelas is None:
        args.kelas = choose_class_interactive()

        if args.kelas is None:
            print("Pilihan kelas tidak valid. Program dihentikan.")
            return

    project_root = Path(__file__).resolve().parents[1]

    model_path = project_root / "models" / f"face_recognition_mobilenet_{args.kelas.lower()}.h5"
    class_indices_path = project_root / "models" / f"class_indices_{args.kelas.lower()}.json"

    if not model_path.exists():
        print("Model MobileNet tidak ditemukan:")
        print(model_path)
        return

    if not class_indices_path.exists():
        print("Class indices tidak ditemukan:")
        print(class_indices_path)
        return

    print("=" * 90)
    print("MOBILENET RECOGNITION ONLY")
    print("=" * 90)
    print(f"Kelas dipilih          : {args.kelas}")
    print(f"Model                  : {model_path}")
    print(f"Recognition threshold  : {args.recognition_threshold}")
    print(f"Recognition frames     : {args.recognition_frames}")
    print(f"Liveness               : OFF")
    print(f"Anti-spoofing          : OFF")
    print("=" * 90)

    model = load_model(str(model_path))
    index_to_class = load_class_indices(class_indices_path)

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print("Webcam tidak terbaca.")
        return

    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )

    recognition_buffer = []
    final_result = "-"
    final_confidence = 0.0
    top3_text = "Top3: -"
    state = "COLLECTING"

    print("\nProgram berjalan.")
    print("r = reset recognition | q = keluar")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        height, width = frame.shape[:2]

        status_text = "Arahkan wajah ke kamera"
        status_color = (0, 0, 255)

        face_box = detect_face_box_mediapipe(frame, face_detector)

        if face_box is None:
            recognition_buffer = []
            final_result = "-"
            final_confidence = 0.0
            top3_text = "Top3: -"
            state = "NO_FACE"
            status_text = "Tidak ada wajah terdeteksi"
            status_color = (0, 0, 255)

        else:
            x, y, fw, fh = face_box
            face_ratio = (fw * fh) / float(width * height)

            cv2.rectangle(display, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

            crop, crop_box = square_crop(
                frame,
                face_box,
                scale=args.crop_scale,
                y_shift=args.crop_y_shift
            )

            if crop_box is not None:
                x1, y1, x2, y2 = crop_box
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if face_ratio < args.min_face_ratio:
                recognition_buffer = []
                state = "TOO_FAR"
                status_text = "Wajah terlalu jauh"
                status_color = (0, 0, 255)

            elif face_ratio > args.max_face_ratio:
                recognition_buffer = []
                state = "TOO_CLOSE"
                status_text = "Wajah terlalu dekat"
                status_color = (0, 0, 255)

            else:
                if state != "DONE":
                    state = "RECOGNITION"

                    probs = predict_recognition_probs(
                        model=model,
                        face_crop=crop,
                        clahe_level=args.clahe_level
                    )

                    recognition_buffer.append(probs)
                    avg_probs = np.mean(np.array(recognition_buffer), axis=0)

                    best_idx = int(np.argmax(avg_probs))
                    best_conf = float(avg_probs[best_idx])
                    best_label = index_to_class.get(best_idx, "UNKNOWN")

                    top3_text = "Top3: " + format_top3(avg_probs, index_to_class)

                    status_text = f"Recognition frame: {len(recognition_buffer)}/{args.recognition_frames}"
                    status_color = (255, 255, 0)

                    if len(recognition_buffer) >= args.recognition_frames:
                        final_confidence = best_conf

                        if best_conf >= args.recognition_threshold:
                            final_result = best_label
                        else:
                            final_result = "Tidak dikenal"

                        state = "DONE"

                else:
                    if final_result == "Tidak dikenal":
                        status_text = f"Tidak dikenal ({final_confidence:.2f})"
                        status_color = (0, 0, 255)
                    else:
                        status_text = f"Dikenali: {final_result} ({final_confidence:.2f})"
                        status_color = (0, 255, 0)

        cv2.putText(display, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.70, status_color, 2)
        cv2.putText(display, f"State: {state}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(display, f"Kelas: {args.kelas}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(display, "Liveness: OFF | Anti-spoofing: OFF", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(display, f"Threshold: {args.recognition_threshold:.2f} | Frames: {len(recognition_buffer)}/{args.recognition_frames}", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
        cv2.putText(display, f"Crop scale: {args.crop_scale:.2f} | CLAHE: {args.clahe_level}", (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
        cv2.putText(display, top3_text, (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.putText(display, "r: reset | q: keluar", (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("MobileNet Recognition Only", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            recognition_buffer = []
            final_result = "-"
            final_confidence = 0.0
            top3_text = "Top3: -"
            state = "COLLECTING"
            print("Reset recognition.")

    cap.release()
    cv2.destroyAllWindows()
    face_detector.close()


if __name__ == "__main__":
    main()
