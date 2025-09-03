# 🚗 Accident Detection System using YOLO, Kalman Filter & Multi-Object Tracking

An advanced real-time vehicle tracking and accident detection system that leverages:
- **YOLO (ONNX model)** for object detection,
- **Kalman Filters** for motion prediction,
- **Hungarian Algorithm** for object matching,
- **Speed Estimation**, **Bounding Box Overlap**, **Class Change Detection**, and **Sudden Stop Detection** for **Accident Prediction**.

---

## 📌 Features

- ✅ Real-time detection and tracking of traffic objects.
- ✅ Speed estimation using Kalman filter motion history.
- ✅ Accident detection using multiple conditions:
  - Bounding box overlap.
  - Relative speed proximity.
  - Center proximity.
  - Sudden drop in vehicle speed.
  - Class label inconsistencies.
- ✅ Stabilized class labels to prevent misclassification.
- ✅ Clear visualization of accidents and possible accidents.
- ✅ Logging support for debugging and analysis.

---

## 📁 Directory Structure

```
accident_detection/
├── main.py
├── detector.py
├── multitracker.py
├── track.py
├── kalman_filter.py
├── accident_detector.py
├── utils.py
├── hungarian_matcher.py
├── models/
│   └── yolov7_5classes_640x640.onnx
├── data/
│   └── traffic.names
├── videos/
│   └── accident2.mp4
└── output/
    ├── accident_output.avi
    └── logs/
        ├── main_log.txt
        ├── track_log.txt
        ├── multitracker_log.txt
        └── accident_log.txt
```

---

## ⚙ Requirements

- Python 3.8+
- OpenCV
- NumPy
- ONNX Runtime

```bash
pip install opencv-python numpy onnxruntime
```

## 📦 Model Files Setup

Due to GitHub's file size limitations (100MB per file), the YOLO model files are not included in this repository. You need to download them separately:

### Required Models:
1. **custom2_1280.onnx** (139.59 MB) - Custom trained model
2. **yolov7_5classes_640x640.onnx** (139.36 MB) - YOLOv7 5-class model  
3. **working.onnx** - Working model file

### Download Instructions:
1. Create a `models/` folder in your project directory if it doesn't exist
2. Download the model files from one of these sources:
   - **Google Drive**: https://drive.google.com/drive/folders/16gM9xko2g0lbqjx2yg1MDF3Q-vojON2q?usp=sharing
3. Place all `.onnx` files in the `models/` folder

### Folder Structure After Download:
```
models/
├── custom2_1280.onnx
├── yolov7_5classes_640x640.onnx
└── working.onnx
```

⚠️ **Important**: The system will not work without these model files!

---

## ▶ How to Run

```
python main.py
```

**Output:** Tracked video with visual overlays showing detected tracks, speed, and accidents.

---

## 🧠 Accident Detection Logic

Accidents are detected based on a multi-condition scoring system. An accident is confirmed if enough conditions are satisfied for a fixed number of frames.

### 🚩 Detection Criteria

Each matched pair accumulates a **score** and must exceed a `score_threshold` to be confirmed.

| **Criteria**             | **Description**                                         |
|--------------------------|---------------------------------------------------------|
| **Bounding Box Overlap** | Boxes of two objects intersect                          |
| **Low Speed**            | Both vehicles below `speed_threshold`                   |
| **Relative Speed**       | Speed difference is low (collision likelihood)          |
| **Center Proximity**     | Distance between object centers is below threshold      |
| **Sudden Stop**          | Speed drops from high to near zero                      |
| **Class Change**         | Rapid object class change after stabilization           |

---

## ⚙ Key Configuration Parameters

These can be set inside `main.py` → `AccidentDetector()` instantiation:

| Parameter                    | Purpose                                     | Default |
|------------------------------|---------------------------------------------|---------|
| `speed_threshold`            | Low-speed cutoff for accident detection     | 6.0 mph |
| `overlap_threshold`          | IOU threshold between bounding boxes        | 0.3     |
| `relative_speed_threshold`   | Relative speed threshold                    | 6.0     |
| `center_proximity_threshold` | Center distance threshold (in pixels)       | 30      |
| `confirm_frames`             | Frames required to confirm accident         | 5       |
| `accident_display_duration`  | Duration to display accident box            | 40      |
| `speed_drop_threshold`       | Threshold for identifying sudden stops      | 70      |
| `possible_accident_threshold`| For identifying possible stops              | 50      |
| `speed_drop_confirm_frames`  | Frames to confirm sustained stop            | 5       |
| `class_freeze_frames`        | Frames used to stabilize class ID           | 10      |

---

## 📝 Customization

- **Change Video Input:** Replace `videos/accident2.mp4` with your custom video.
- **Model Update:** Replace `models/yolov7_5classes_640x640.onnx` with a fine-tuned YOLO model.
- **Detection Classes:** Update `data/traffic.names` as per your dataset.
- **Visualization:** Modify `draw_detections()` in `utils.py` to add/remove overlays.

---

## 📊 Logs & Debugging

All internal events are logged in:
- `output/logs/main_log.txt`
- `output/logs/track_log.txt`
- `output/logs/accident_log.txt`
- `output/logs/multitracker_log.txt`

These help in detailed debugging, analyzing frame-by-frame behavior and accident triggers.

---

## 📌 Limitations & Future Scope

- Slight camera angle artifacts may cause false positives.
- False classification during occlusion can affect detection (partially solved by class freezing).
- Stationary object detection still relies on thresholds — further camera calibration can improve accuracy.

Future enhancements:
- Lane mapping
- Scene understanding
- Collision direction inference
- Real-world speed calibration via perspective transforms

---

## ❤️ Credits

Developed with dedication and countless iterations to handle real-world accident detection challenges.  
Thanks to the [YOLO](https://github.com/WongKinYiu/yolov7) and [ONNX Runtime](https://onnxruntime.ai/) communities.

---

## 📬 Contact

For questions or suggestions, feel free to reach out!