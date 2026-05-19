import json
import math
import os
import time

import cv2
import numpy as np
import torch
import open3d as o3d
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

CALIBRATION_PATH = "calibration_dpt.json"
MIDAS_MODEL_NAME = "DPT_Hybrid"
FACE_MODEL_PATH = "blaze_face_short_range.tflite"

FOV_DEGREES = 70.0
MAX_DEPTH_M = 10.0
VOXEL_SIZE_M = 0.01
FACE_SPHERE_RADIUS_M = 0.1
COORD_AXIS_SIZE_M = 0.5

REAL_INTEROCULAR_M = 0.063

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
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
print(f"MiDaS ready on {device}.")

def run_midas(rgb_frame):
    h, w = rgb_frame.shape[:2]
    input_batch = transform(rgb_frame).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()

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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam capture resolution: {actual_w}x{actual_h}")

scale = load_calibration()
if scale is None:
    print("No calibration found. The first snapshot with a face visible will "
          "auto-calibrate using eye-width.")
else:
    print(f"Loaded calibration: scale={scale:.4f}")
print("Webcam opened. Press 's' to snapshot, 'c' to recalibrate manually, 'q' to quit.")

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

def unproject(depth_m, rgb_frame, fx, fy, cx, cy, max_depth):
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    X = (u - cx) * depth_m / fx
    Y = -((v - cy) * depth_m / fy)
    Z = -depth_m

    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    colors = rgb_frame.reshape(-1, 3).astype(np.float64) / 255.0

    mask = np.isfinite(points).all(axis=1) & (points[:, 2] < -0.01)
    if max_depth is not None:
        mask &= points[:, 2] > -max_depth
    return points[mask], colors[mask]

def unproject_pixel(u, v, depth_m, fx, fy, cx, cy):
    X = (u - cx) * depth_m / fx
    Y = -((v - cy) * depth_m / fy)
    Z = -depth_m
    return [float(X), float(Y), float(Z)]

def show_viewer(pcd, axis, face_geos):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="RoomScan Stage 4 - 3D Viewer")

    render_opt = vis.get_render_option()
    render_opt.point_size = 4.0

    vis.add_geometry(pcd)
    vis.add_geometry(axis)
    for s in face_geos:
        vis.add_geometry(s)

    scene_center = (pcd.get_center().tolist() if len(pcd.points) > 0
                    else [0.0, 0.0, -1.5])

    ctr = vis.get_view_control()
    ctr.set_lookat(scene_center)
    ctr.set_front([0.0, 0.0, 1.0])
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_zoom(0.5)

    faces_visible = [True]
    axis_visible = [True]

    def toggle_faces(v):
        if not face_geos:
            return False
        if faces_visible[0]:
            for s in face_geos:
                v.remove_geometry(s, reset_bounding_box=False)
        else:
            for s in face_geos:
                v.add_geometry(s, reset_bounding_box=False)
        faces_visible[0] = not faces_visible[0]
        return False

    def toggle_axis(v):
        if axis_visible[0]:
            v.remove_geometry(axis, reset_bounding_box=False)
        else:
            v.add_geometry(axis, reset_bounding_box=False)
        axis_visible[0] = not axis_visible[0]
        return False

    def focus(point, zoom, front, up):
        def cb(v):
            c = v.get_view_control()
            c.set_lookat(list(point))
            c.set_front(list(front))
            c.set_up(list(up))
            c.set_zoom(zoom)
            return False
        return cb

    def make_face_focus(idx):
        def cb(v):
            if idx >= len(face_geos):
                print(f"No face {idx + 1}.")
                return False
            c = v.get_view_control()
            c.set_lookat(face_geos[idx].get_center().tolist())
            c.set_front([0.0, 0.0, 1.0])
            c.set_up([0.0, 1.0, 0.0])
            c.set_zoom(0.15)
            return False
        return cb

    backgrounds = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.10, 0.10, 0.15]),
        np.array([0.5, 0.5, 0.5]),
    ]
    bg_idx = [0]

    def cycle_background(v):
        bg_idx[0] = (bg_idx[0] + 1) % len(backgrounds)
        v.get_render_option().background_color = backgrounds[bg_idx[0]]
        return False

    def bigger_points(v):
        opt = v.get_render_option()
        opt.point_size = min(opt.point_size + 1, 20)
        return False

    def smaller_points(v):
        opt = v.get_render_option()
        opt.point_size = max(opt.point_size - 1, 1)
        return False

    vis.register_key_callback(ord("H"), toggle_faces)
    vis.register_key_callback(ord("A"), toggle_axis)
    vis.register_key_callback(ord("F"),
        focus(scene_center, 0.5, [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]))
    vis.register_key_callback(ord("O"),
        focus([0.0, 0.0, 0.0], 0.3, [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]))
    vis.register_key_callback(ord("T"),
        focus(scene_center, 0.5, [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]))
    vis.register_key_callback(ord("R"),
        focus(scene_center, 0.5, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
    vis.register_key_callback(ord("B"), cycle_background)
    vis.register_key_callback(ord("]"), bigger_points)
    vis.register_key_callback(ord("["), smaller_points)

    for i in range(min(9, len(face_geos))):
        vis.register_key_callback(ord(str(i + 1)), make_face_focus(i))

    print("3D viewer controls:")
    print("  mouse drag = orbit | shift+drag = pan | scroll = zoom")
    if face_geos:
        n = min(9, len(face_geos))
        print(f"  1-{n} = focus on face N    H = toggle red face markers")
    print("  A = toggle XYZ axis triad")
    print("  F = frame whole scene (front)")
    print("  T = top-down view")
    print("  R = right-side view")
    print("  O = look at camera origin")
    print("  B = cycle background color")
    print("  [ / ] = shrink / grow point size")
    print("  Q / Esc = close viewer")

    vis.run()
    vis.destroy_window()

def take_snapshot(rgb_frame, detections, scale):
    h, w = rgb_frame.shape[:2]
    fx = (w / 2) / math.tan(math.radians(FOV_DEGREES / 2))
    fy = fx
    cx, cy = w / 2, h / 2

    print("Running MiDaS for snapshot...")
    midas_depth = run_midas(rgb_frame)

    if scale is None and detections:
        for det in detections:
            eye_dist_m = estimate_distance_from_eyes(det, w, h, fx)
            if eye_dist_m is None:
                continue
            bbox = det.bounding_box
            fu = int(max(0, min(w - 1, bbox.origin_x + bbox.width // 2)))
            fv = int(max(0, min(h - 1, bbox.origin_y + bbox.height // 2)))
            midas_val = float(midas_depth[fv, fu])
            if midas_val <= 1e-6:
                continue
            scale = eye_dist_m * midas_val
            save_calibration(scale)
            print(f"Auto-calibrated from eye-width: face was ~{eye_dist_m:.2f}m away.")
            break
        if scale is None:
            print("Auto-cal failed (no usable face). Cloud will use arbitrary units.")

    calibrated = scale is not None
    scale_factor = scale if calibrated else 1.0

    midas_safe = np.maximum(midas_depth, 1e-6)
    depth_m = scale_factor / midas_safe

    max_d = MAX_DEPTH_M if calibrated else None
    points, colors = unproject(depth_m, rgb_frame, fx, fy, cx, cy, max_d)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if calibrated:
        pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE_M)
        if len(pcd.points) > 50:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0)

    face_geos = []
    for det in detections:
        bbox = det.bounding_box
        fu = int(max(0, min(w - 1, bbox.origin_x + bbox.width // 2)))
        fv = int(max(0, min(h - 1, bbox.origin_y + bbox.height // 2)))
        midas_val = float(midas_depth[fv, fu])
        if midas_val <= 1e-6:
            continue
        d_m = scale_factor / midas_val
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=FACE_SPHERE_RADIUS_M)
        sphere.translate(unproject_pixel(fu, fv, d_m, fx, fy, cx, cy))
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        sphere.compute_vertex_normals()
        face_geos.append(sphere)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=COORD_AXIS_SIZE_M)

    units = "meters" if calibrated else "arbitrary units"
    print(f"Snapshot: {len(pcd.points)} points ({units}), "
          f"{len(face_geos)} face markers.")
    print("Close the 3D viewer to return to the live camera.")

    show_viewer(pcd, axis, face_geos)

    return scale


start = time.monotonic()
last_timestamp_ms = -1
last_rgb = None
last_detections = []

while True:
    success, frame = cap.read()
    if not success:
        print("Lost connection to webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    last_rgb = rgb_frame

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.monotonic() - start) * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = timestamp_ms
    result = face_detector.detect_for_video(mp_image, timestamp_ms)
    last_detections = result.detections or []

    h_img, w_img = frame.shape[:2]
    fx_px = (w_img / 2) / math.tan(math.radians(FOV_DEGREES / 2))

    for det in last_detections:
        bbox = det.bounding_box
        x, y = bbox.origin_x, bbox.origin_y
        bw, bh = bbox.width, bbox.height
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        confidence = int(det.categories[0].score * 100)

        eye_dist_m = estimate_distance_from_eyes(det, w_img, h_img, fx_px)
        if eye_dist_m is not None:
            top_label = f"{eye_dist_m:.2f}m (eye)"
        else:
            top_label = f"Face {confidence}%"
        cv2.putText(frame, top_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if scale is not None:
        status = f"MiDaS calibrated (scale={scale:.4f})"
        status_color = (0, 255, 255)
    else:
        status = "MiDaS not calibrated - press 's' to snapshot + auto-cal"
        status_color = (0, 165, 255)
    cv2.putText(frame, status, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    cv2.putText(frame, "s: 3D snapshot   c: manual cal   q: quit",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("RoomScan Stage 4 - Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if last_rgb is None:
            continue
        scale = take_snapshot(last_rgb, last_detections, scale)
    elif key == ord('c'):
        if not last_detections:
            print("Calibration: no face visible. Try again.")
            continue
        cal_depth = run_midas(last_rgb)
        bbox = last_detections[0].bounding_box
        h, w = last_rgb.shape[:2]
        cu = max(0, min(w - 1, bbox.origin_x + bbox.width // 2))
        cv_y = max(0, min(h - 1, bbox.origin_y + bbox.height // 2))
        midas_val = float(cal_depth[cv_y, cu])
        if midas_val <= 1e-6:
            print("Calibration: depth sample invalid. Try again.")
            continue
        try:
            raw = input("Enter your current distance from the camera in meters: ")
            known_m = float(raw)
            if known_m <= 0:
                raise ValueError("must be positive")
            scale = known_m * midas_val
            save_calibration(scale)
        except ValueError as e:
            print(f"Calibration cancelled (bad input: {e}).")

cap.release()
cv2.destroyAllWindows()
face_detector.close()
print("Done.")
