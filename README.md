# AI Vehicle Tracking & Speed Estimation 🚀

An end-to-end computer vision pipeline that detects, tracks, and estimates the real-time speed of oncoming vehicles using **YOLOv8** and **DeepSORT** in Python.

---

## 📌 Features

- **Real-Time Detection:** Powered by the fast and accurate YOLOv8 nano model.
- **Robust Object Tracking:** Utilizes DeepSORT to maintain persistent vehicle IDs across frames, minimizing ID switches.
- **Direction-Aware Speed Estimation:** Tracks down-moving traffic and measures elapsed time between two designated virtual line thresholds.
- **Live Diagnostics:** On-screen display of tracked bounding boxes, object class names, calculated speeds (in km/h), and live frame-rate (FPS).
- **Auto-Video Recording:** Exports processed live feeds smoothly straight into a compressed `output.mp4` video.

---

## 🛠️ Architecture & Workflow

1. **Object Detection:** Every incoming frame from OpenCV's `VideoCapture` is sent through YOLOv8 to extract bounding boxes for target vehicles (`car`, `motorcycle`, `bus`, `truck`).
2. **Feature Tracking:** Extracted bounding boxes are converted to LTRW formats and handed off to DeepSORT, which pairs detection coordinates with deep visual features.
3. **Trajectory & Vector Logic:** The script records the recent center-point (`cx, cy`) history of each ID to ensure the vehicle is traveling downward.
4. **Time-Of-Flight Speed Calculation:** 
   - A timer initiates when the centroid crosses the entry boundary line (`line1_y`).
   - The timer halts as the centroid breaches the exit boundary line (`line2_y`).
   - Speed is calculated using the formula:
     $$\text{Speed (km/h)} = \left( \frac{\text{Distance (meters)}}{\text{Time Taken (seconds)}} \right) \times 3.6$$

---

## 💻 Installation & Requirements

### 1. Prerequisites
Ensure you have **Python 3.8 or higher** installed.

### 2. Dependency Installation
Install all required libraries using `pip`:

```bash
pip install opencv-python numpy ultralytics deep-sort-realtime
