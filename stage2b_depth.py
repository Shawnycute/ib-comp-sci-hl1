import json
import os
import time

import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

FRAME_SKIP_N = 3
CALIBRATION_PATH = "calibration.json"
MIDAS_MODEL_NAME = "MiDaS_small"
FACE_MODEL_PATH = "blaze_face_short_range.tflite"

def load_calibration():
    if not os.path.exists(CALIBRATION_PATH):
        return None
    try:
        with open(CALIBRATION_PATH, "r") as f:
            data = json.load(f)
        return float(data["scale"])
    except (ValueError, KeyError, OSError):
        print("WARNING: calibration.json is unreadable, ignoring.")
        return None

def save_calibration(scale):
    with open(CALIBRATION_PATH, "w") as f:
        json.dump({"scale": scale}, f)
    print(f"Saved calibration (scale={scale:.4f}) to {CALIBRATION_PATH}.")

print("Loading MiDaS... (first run downloads weights)")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_NAME)
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform
print(f"MiDaS ready on {device}.")

options = vision.FaceDetectorOptions(
    base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    min_detection_confidence=0.5,
)
face_detector = vision.FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam. Is another app using it?")
    exit()

scale = load_calibration()
if scale is None:
    print("No calibration found. Press 'c' with a face visible to calibrate.")
else:
    print(f"Loaded calibration: scale={scale:.4f}")

print("Webcam opened. Press 'q' to quit, 'c' to calibrate.")

start = time.monotonic()
last_timestamp_ms = -1
frame_idx = 0
latest_depth = None
latest_depth_vis = None

while True:
    success, frame = cap.read()
    if not success:
        print("Lost connection to webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.monotonic() - start) * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = timestamp_ms
    result = face_detector.detect_for_video(mp_image, timestamp_ms)

    if frame_idx % FRAME_SKIP_N == 0:
        input_batch = transform(rgb_frame).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        latest_depth = prediction.cpu().numpy()

        depth_norm = cv2.normalize(latest_depth, None, 0, 255,
                                   cv2.NORM_MINMAX).astype(np.uint8)
        latest_depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
    frame_idx += 1

    face_center_depth = None
    if result.detections:
        for detection in result.detections:
            bbox = detection.bounding_box
            x, y = bbox.origin_x, bbox.origin_y
            box_w, box_h = bbox.width, bbox.height

            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h),
                          (0, 255, 0), 2)

            distance_label = None
            if latest_depth is not None:
                cx = max(0, min(w - 1, x + box_w // 2))
                cy = max(0, min(h - 1, y + box_h // 2))
                midas_val = float(latest_depth[cy, cx])
                face_center_depth = midas_val

                if scale is not None and midas_val > 1e-6:
                    distance_m = scale / midas_val
                    distance_label = f"{distance_m:.1f}m"

            confidence = int(detection.categories[0].score * 100)
            top_label = f"Face {confidence}%"
            if distance_label:
                top_label += f"  {distance_label}"
            cv2.putText(frame, top_label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if latest_depth_vis is not None:
                cv2.rectangle(latest_depth_vis, (x, y),
                              (x + box_w, y + box_h), (0, 255, 0), 2)
                if distance_label:
                    cv2.putText(latest_depth_vis, distance_label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)

    cv2.imshow("RoomScan Stage 2b - Camera", frame)
    if latest_depth_vis is not None:
        cv2.imshow("RoomScan Stage 2b - Depth", latest_depth_vis)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        if face_center_depth is None or face_center_depth <= 1e-6:
            print("Calibration: no face/depth sample available. Try again.")
        else:
            try:
                raw = input("Enter your current distance from the camera in meters: ")
                known_m = float(raw)
                if known_m <= 0:
                    raise ValueError("must be positive")
                scale = known_m * face_center_depth
                save_calibration(scale)
            except ValueError as e:
                print(f"Calibration cancelled (bad input: {e}).")

cap.release()
cv2.destroyAllWindows()
face_detector.close()
print("Done.")
