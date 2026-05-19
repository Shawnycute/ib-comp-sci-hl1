import json
import math
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

FOV_DEGREES = 70.0
PIXELS_PER_METER = 100
RADAR_SIZE = 600
CAMERA_X = RADAR_SIZE // 2
CAMERA_Y = RADAR_SIZE - 20
MAX_RADAR_RANGE_M = 5

TRACK_MATCH_THRESHOLD_PX = 120
TRACK_MAX_MISSED_FRAMES = 30

REAL_INTEROCULAR_M = 0.063
AUTO_CAL_BUFFER = 30
AUTO_CAL_MIN_SAMPLES = 10

PERSON_COLORS = [
    (0, 255, 0),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (180, 105, 255),
]

def load_calibration():
    if not os.path.exists(CALIBRATION_PATH):
        return None
    try:
        with open(CALIBRATION_PATH, "r") as f:
            return float(json.load(f)["scale"])
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
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
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
    print("No calibration found. Eye-width auto-calibration will run when a face is visible.")
else:
    print(f"Loaded calibration: scale={scale:.4f}")
print("Webcam opened. Press 'q' to quit, 'c' to recalibrate manually.")

def estimate_distance_from_eyes(detection, w, h, fx_px):
    kps = getattr(detection, "keypoints", None)
    if not kps or len(kps) < 2:
        return None
    eye_a, eye_b = kps[0], kps[1]
    dx = (eye_a.x - eye_b.x) * w
    dy = (eye_a.y - eye_b.y) * h
    pixel_dist = math.hypot(dx, dy)
    if pixel_dist < 2:
        return None
    return (REAL_INTEROCULAR_M * fx_px) / pixel_dist

def world_to_radar(distance_m, angle_rad):
    forward_px = distance_m * math.cos(angle_rad) * PIXELS_PER_METER
    sideways_px = distance_m * math.sin(angle_rad) * PIXELS_PER_METER
    return int(CAMERA_X + sideways_px), int(CAMERA_Y - forward_px)

def draw_radar(people):
    radar = np.zeros((RADAR_SIZE, RADAR_SIZE, 3), dtype=np.uint8)

    for r_m in range(1, MAX_RADAR_RANGE_M + 1):
        radius_px = r_m * PIXELS_PER_METER
        cv2.circle(radar, (CAMERA_X, CAMERA_Y), radius_px, (50, 50, 50), 1)
        cv2.putText(radar, f"{r_m}m",
                    (CAMERA_X + 4, CAMERA_Y - radius_px + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1)

    half_fov_rad = math.radians(FOV_DEGREES / 2)
    left_end = world_to_radar(MAX_RADAR_RANGE_M, -half_fov_rad)
    right_end = world_to_radar(MAX_RADAR_RANGE_M, half_fov_rad)
    cv2.line(radar, (CAMERA_X, CAMERA_Y), left_end, (60, 60, 90), 1)
    cv2.line(radar, (CAMERA_X, CAMERA_Y), right_end, (60, 60, 90), 1)

    cam_tri = np.array([
        [CAMERA_X, CAMERA_Y - 12],
        [CAMERA_X - 10, CAMERA_Y + 8],
        [CAMERA_X + 10, CAMERA_Y + 8],
    ], dtype=np.int32)
    cv2.fillPoly(radar, [cam_tri], (0, 200, 0))

    cv2.arrowedLine(radar, (30, 50), (30, 20), (200, 200, 200), 2, tipLength=0.4)
    cv2.putText(radar, "FWD", (12, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    for p in people:
        color = PERSON_COLORS[(p["id"] - 1) % len(PERSON_COLORS)]
        cv2.circle(radar, (p["x"], p["y"]), 7, color, -1)
        cv2.circle(radar, (p["x"], p["y"]), 8, (255, 255, 255), 1)
        cv2.putText(radar, f"P{p['id']}: {p['distance_m']:.1f}m",
                    (p["x"] + 10, p["y"] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return radar

tracked = []
next_id = 1

def update_tracker(detections):
    global next_id
    matched_ids = set()
    for det in detections:
        best, best_dist = None, float("inf")
        for t in tracked:
            if t["id"] in matched_ids:
                continue
            d = math.hypot(t["x"] - det["x"], t["y"] - det["y"])
            if d < best_dist:
                best, best_dist = t, d
        if best is not None and best_dist < TRACK_MATCH_THRESHOLD_PX:
            best["x"], best["y"], best["missed"] = det["x"], det["y"], 0
            det["id"] = best["id"]
            matched_ids.add(best["id"])
        else:
            det["id"] = next_id
            tracked.append({"id": next_id, "x": det["x"], "y": det["y"], "missed": 0})
            matched_ids.add(next_id)
            next_id += 1
    for t in tracked:
        if t["id"] not in matched_ids:
            t["missed"] += 1
    tracked[:] = [t for t in tracked if t["missed"] < TRACK_MAX_MISSED_FRAMES]

start = time.monotonic()
last_timestamp_ms = -1
frame_idx = 0
latest_depth = None
latest_depth_vis = None
auto_scale_samples = []
auto_cal_written = scale is not None

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

    fx_px = (w / 2) / math.tan(math.radians(FOV_DEGREES / 2))

    detections = []
    calibration_depth_sample = None
    if result.detections:
        for d in result.detections:
            bbox = d.bounding_box
            x, y, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            cx_img = x + bw / 2

            normalized_x = (cx_img - w / 2) / (w / 2)
            angle_deg = normalized_x * (FOV_DEGREES / 2)
            angle_rad = math.radians(angle_deg)

            eye_distance_m = estimate_distance_from_eyes(d, w, h, fx_px)

            midas_val = None
            if latest_depth is not None:
                cx = max(0, min(w - 1, x + bw // 2))
                cy = max(0, min(h - 1, y + bh // 2))
                midas_val = float(latest_depth[cy, cx])
                if calibration_depth_sample is None and midas_val > 1e-6:
                    calibration_depth_sample = midas_val

            if eye_distance_m is not None:
                distance_m, source = eye_distance_m, "eye"
            elif midas_val is not None and midas_val > 1e-6 and scale is not None:
                distance_m, source = scale / midas_val, "midas"
            else:
                distance_m, source = None, None

            if (eye_distance_m is not None
                    and midas_val is not None and midas_val > 1e-6):
                auto_scale_samples.append(eye_distance_m * midas_val)
                if len(auto_scale_samples) > AUTO_CAL_BUFFER:
                    auto_scale_samples.pop(0)
                if (not auto_cal_written
                        and len(auto_scale_samples) >= AUTO_CAL_MIN_SAMPLES):
                    scale = float(np.median(auto_scale_samples))
                    save_calibration(scale)
                    auto_cal_written = True
                    print(f"Auto-calibrated MiDaS from "
                          f"{len(auto_scale_samples)} eye samples.")

            det_record = {
                "bbox": (x, y, bw, bh),
                "confidence": int(d.categories[0].score * 100),
                "distance_m": distance_m,
                "midas_val": midas_val,
                "source": source,
                "angle_deg": angle_deg,
                "angle_rad": angle_rad,
                "id": None,
            }
            if distance_m is not None:
                rx, ry = world_to_radar(distance_m, angle_rad)
                det_record["x"], det_record["y"] = rx, ry
            detections.append(det_record)

    trackable = [d for d in detections if d["distance_m"] is not None]
    update_tracker(trackable)

    for det in detections:
        x, y, bw, bh = det["bbox"]
        color = ((0, 255, 0) if det["id"] is None
                 else PERSON_COLORS[(det["id"] - 1) % len(PERSON_COLORS)])

        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

        if det["id"] is not None:
            top_label = (f"P{det['id']}: {det['distance_m']:.2f}m "
                         f"({det['source']})")
        elif det["distance_m"] is not None:
            top_label = (f"{det['distance_m']:.2f}m ({det['source']})")
        else:
            top_label = f"Face {det['confidence']}%"
        cv2.putText(frame, top_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        midas_str = (f"midas={det['midas_val']:.3f}"
                     if det["midas_val"] is not None else "midas:n/a")
        cv2.putText(frame,
                    f"{midas_str}  ang={det['angle_deg']:+.0f}deg",
                    (x, y + bh + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        if latest_depth_vis is not None:
            cv2.rectangle(latest_depth_vis, (x, y), (x + bw, y + bh), color, 2)
            if det["id"] is not None:
                cv2.putText(latest_depth_vis,
                            f"P{det['id']}: {det['distance_m']:.1f}m",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

    radar_img = draw_radar(trackable)

    if scale is not None:
        hud1 = f"MiDaS CALIBRATED  scale={scale:.4f}"
        hud1_color = (0, 255, 255)
    else:
        hud1 = (f"MiDaS warming up  ({len(auto_scale_samples)}/"
                f"{AUTO_CAL_MIN_SAMPLES} eye samples)")
        hud1_color = (0, 165, 255)
    depth_state = ("depth: ready" if latest_depth is not None
                   else "depth: warming up...")
    hud2 = (f"faces={len(detections)}  tracked={len(trackable)}  "
            f"{depth_state}  FOV={FOV_DEGREES:.0f}deg "
            f"fx={(w/2)/math.tan(math.radians(FOV_DEGREES/2)):.0f}px")
    cv2.putText(frame, hud1, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud1_color, 1)
    cv2.putText(frame, hud2, (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame,
                "labels show (eye) = auto from eye-width, "
                "(midas) = from MiDaS+scale",
                (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cv2.putText(frame, "c: re-calibrate manually   q: quit", (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("RoomScan Stage 3 - Camera", frame)
    if latest_depth_vis is not None:
        cv2.imshow("RoomScan Stage 3 - Depth", latest_depth_vis)
    cv2.imshow("RoomScan Stage 3 - Top-Down View", radar_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        if calibration_depth_sample is None:
            print("Calibration: no face/depth sample available. Try again.")
        else:
            try:
                raw = input("Enter your current distance from the camera in meters: ")
                known_m = float(raw)
                if known_m <= 0:
                    raise ValueError("must be positive")
                scale = known_m * calibration_depth_sample
                save_calibration(scale)
                auto_cal_written = True
            except ValueError as e:
                print(f"Calibration cancelled (bad input: {e}).")

cap.release()
cv2.destroyAllWindows()
face_detector.close()
print("Done.")
