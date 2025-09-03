# Accident Detection and Object Tracking System

A modular Python implementation of object detection and multi-object tracking using YOLOv7 (ONNX), Kalman Filter, and Hungarian Assignment.

---

## 📁 Folder Structure

project/ ├── detector.py ├── kalman_filter.py ├── track.py ├── utils.py ├── main.py ├── models/ │ └── custom2_1280.onnx └── README.md

yaml
Copy
Edit

---

## ⚙️ Requirements

Install the required packages:

```bash
pip install opencv-python numpy onnxruntime
If you want GPU acceleration (optional, requires compatible hardware + drivers):

bash
Copy
Edit
pip install onnxruntime-gpu
🧠 What This Project Does
Runs YOLOv7 ONNX model to detect objects (car, bus, person, cyclist, etc.)
Tracks objects using Kalman Filter + Hungarian Algorithm
Displays real-time bounding boxes and trajectories
Modular design for easy maintenance and future upgrades
▶️ Running the Code
bash
Copy
Edit
python main.py
By default, it uses your webcam.
To test with a video file instead, open main.py and replace:

python
Copy
Edit
cap = cv2.VideoCapture(0)
with:

python
Copy
Edit
cap = cv2.VideoCapture("sample_video.mp4")
📦 Model Used
The model used is a YOLOv7 ONNX export.

Place the .onnx file in the models/ folder:

bash
Copy
Edit
models/custom2_1280.onnx
You can convert your own PyTorch YOLO model using export tools (ask if you want help with that).

📚 Modules Explained
File	Description
detector.py	Loads and runs the YOLOv7 ONNX model
kalman_filter.py	Kalman filter-based motion prediction
track.py	Multi-object tracking, matching, and assignment
utils.py	Helper functions like drawing boxes and traces
main.py	Main execution loop (video capture, detection, tracking, display)
🚀 Future Scope
Add speed estimation
Add accident/impact/conflict detection logic
Add alert or notification system
Build a frontend web/mobile app for visualization
💬 Credits
Based on modular adaptation of a C++-based accident detection research project.

🤝 Contributing
Feel free to fork and enhance this modular system!

yaml
Copy
Edit

---

Would you like me to also give you:
- A `.gitignore` file?
- Example test video links?
- Code comments for newcomers?

Let me know, I’ll include them!