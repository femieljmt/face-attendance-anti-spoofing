
import cv2
import json
import argparse
from pathlib import Path

import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model


def detect_face_box(frame_bgr, detector):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

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


def square_crop(image, box, scale=1.25):
    x, y, w, h = box
    ih, iw = image.shape[:2]

    cx = x + w / 2
    cy = y + h / 2
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
    if x2 > iw:
        x1 -= (x2 - iw)
        x2 = iw
    if y2 > ih:
        y1 -= (y2 - ih)
        y2 = ih

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(iw, x2)
    y2 = min(ih, y2)

    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def predict_spoof_probability(model, crop, img_size=224):
    img = cv2.resize(crop, (img_size, img_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = img_rgb.astype("float32")
    arr = np.expand_dims(arr, axis=0)
    prob = float(model.predict(arr, verbose=0)[0][0])
    return prob


def main():
    parser = argparse.ArgumentParser(description="Realtime test anti-spoofing MobileNet V2 real vs spoof.")
    parser.add_argument("--model_path", type=str, default="models/anti_spoofing_mobilenet_real_spoof.h5")
    parser.add_argument("--threshold_path", type=str, default="models/anti_spoofing_threshold.json")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--crop_scale", type=float, default=1.25)
    parser.add_argument("--det_conf", type=float, default=0.35)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    threshold_path = Path(args.threshold_path)

    if not model_path.exists():
        print("Model tidak ditemukan:", model_path)
        return

    threshold = 0.50
    if threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as f:
            threshold = float(json.load(f).get("threshold", 0.50))

    if args.threshold is not None:
        threshold = args.threshold

    print("=" * 80)
    print("REALTIME ANTI-SPOOFING TEST V2")
    print("=" * 80)
    print("Model:", model_path)
    print("Threshold:", threshold)
    print("Arti output: prob >= threshold berarti SPOOF")
    print("q=keluar | r=reset")

    model = load_model(str(model_path))

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Kamera tidak terbuka.")
        return

    mp_face_detection = mp.solutions.face_detection
    detector = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=args.det_conf
    )

    buffer = []
    result_text = "Arahkan wajah"
    result_color = (0, 255, 255)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        box = detect_face_box(frame, detector)

        if box is None:
            buffer = []
            result_text = "Tidak ada wajah"
            result_color = (0, 0, 255)
        else:
            x, y, w, h = box
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            crop, crop_box = square_crop(frame, box, scale=args.crop_scale)
            x1, y1, x2, y2 = crop_box
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)

            prob = predict_spoof_probability(model, crop)
            buffer.append(prob)

            if len(buffer) > args.frames:
                buffer.pop(0)

            avg_prob = float(np.mean(buffer))
            if len(buffer) < args.frames:
                result_text = f"Collecting {len(buffer)}/{args.frames} | spoof_prob={avg_prob:.3f}"
                result_color = (255, 255, 0)
            else:
                if avg_prob >= threshold:
                    result_text = f"SPOOF / HP REPLAY ({avg_prob:.3f})"
                    result_color = (0, 0, 255)
                else:
                    result_text = f"REAL ({avg_prob:.3f})"
                    result_color = (0, 255, 0)

        cv2.putText(display, result_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, result_color, 2)
        cv2.putText(display, f"Threshold: {threshold:.3f} | frames: {len(buffer)}/{args.frames}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(display, "q: keluar | r: reset", (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("Anti-spoofing MobileNet Test V2", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            buffer = []
            print("Reset buffer.")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
