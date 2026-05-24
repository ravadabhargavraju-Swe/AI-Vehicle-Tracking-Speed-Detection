import cv2
import numpy as np
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO
model = YOLO("yolov8n.pt")
class_names = model.names

# DeepSORT
tracker = DeepSort(max_age=30)

# Video Capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Video Writer (SAVE OUTPUT)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

# Speed Setup
distance_meters = 5  
line1_y = 200
line2_y = 350

vehicle_times = {}
vehicle_speeds = {}
track_history = {}
prev_time = 0

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # YOLO Detection
    results = model(frame, imgsz=320, conf=0.5, verbose=False)[0]
    detections = []

    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r

        w = x2 - x1
        h = y2 - y1

        detections.append((
            [float(x1), float(y1), float(w), float(h)],
            float(score),
            int(class_id)
        ))

    # Tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    # Process Tracks
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id 

        l, t, r, b = track.to_ltrb()

        class_id = track.det_class if track.det_class is not None else -1
        class_name = class_names[class_id] if class_id != -1 else "object"

        cx = int((l + r) / 2)
        cy = int((t + b) / 2)

        # Track History
        if track_id not in track_history:
            track_history[track_id] = []

        track_history[track_id].append((cx, cy))

        # Limit memory
        if len(track_history[track_id]) > 30:
            track_history[track_id].pop(0)

        # Vehicle Speed Logic
        if class_name in ["car", "motorcycle", "bus", "truck"]:

            moving_down = False

            if len(track_history[track_id]) >= 5:
                if track_history[track_id][-1][1] > track_history[track_id][-5][1]:
                    moving_down = True

            if moving_down:
                # Start timing
                if cy > line1_y and track_id not in vehicle_times:
                    vehicle_times[track_id] = time.time()

                # End timing
                if cy > line2_y and track_id in vehicle_times:
                    time_taken = time.time() - vehicle_times[track_id]

                    if time_taken > 0:
                        speed = (distance_meters / time_taken) * 3.6
                        vehicle_speeds[track_id] = speed

        # Draw Bounding Box
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)

        label = f"{class_name} ID:{track_id}"

        if track_id in vehicle_speeds:
            label += f" {int(vehicle_speeds[track_id])} km/h"

        cv2.putText(frame, label, (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw Lines
    cv2.line(frame, (0, line1_y), (640, line1_y), (0, 255, 255), 2)
    cv2.line(frame, (0, line2_y), (640, line2_y), (0, 0, 255), 2)

    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Show + Save
    cv2.imshow("AI Vehicle Tracking & Speed Detection", frame)
    out.write(frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
