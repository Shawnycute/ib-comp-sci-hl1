import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "blaze_face_short_range.tflite"

options = vision.FaceDetectorOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    min_detection_confidence=0.5,
)
face_detector = vision.FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam. Is another app using it?")
    exit()

print("Webcam opened. Press 'q' to quit.")

start = time.monotonic()
last_timestamp_ms = -1

while True:
    success, frame = cap.read()
    if not success:
        print("Lost connection to webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.monotonic() - start) * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = timestamp_ms
    result = face_detector.detect_for_video(mp_image, timestamp_ms)

    if result.detections:
        for detection in result.detections:
            bbox = detection.bounding_box
            x = bbox.origin_x
            y = bbox.origin_y
            box_w = bbox.width
            box_h = bbox.height

            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)

            confidence = int(detection.categories[0].score * 100)
            label = f"Face {confidence}%"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("RoomScan Stage 1 - Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_detector.close()
print("Done.")
