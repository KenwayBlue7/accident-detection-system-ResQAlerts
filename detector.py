import cv2
import numpy as np
import onnxruntime as ort
import logging
import os

# Setup logging
os.makedirs("output/logs", exist_ok=True)
logging.basicConfig(filename="output/logs/detector_log.txt", level=logging.DEBUG, filemode='w',
                    format='[%(levelname)s] %(message)s')
logger = logging.getLogger("DetectorLogger")


class YOLODetector:
    def __init__(self, model_path, class_names_path, use_gpu=True, input_size=(640, 640)):
        self.input_size = input_size
        self.class_names = self.load_class_names(class_names_path)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        logger.debug("\n[INIT] YOLODetector Initialized")
        logger.debug(f"Model Path: {model_path}")
        logger.debug(f"Classes Loaded: {self.class_names}")
        logger.debug(f"Using Providers: {providers}")
        logger.debug(f"Input Size: {self.input_size}\n")

    def load_class_names(self, path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def detect(self, frame, scale=(1.0, 1.0), pad=(0, 0)):
        logger.debug("[DETECT] Running Detection...")
        input_blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=self.input_size, swapRB=True, crop=False)

        outputs = self.session.run(None, {self.input_name: input_blob})[0]

        detections = []
        logger.debug(f"Model Output Shape: {outputs.shape}")
        logger.debug(f"Scale Factors: {scale}, Padding: {pad}")

        for idx, det in enumerate(outputs):
            x1, y1, x2, y2 = det[1:5]
            cls_id = int(det[5])
            conf = float(det[6])

            label = self.class_names[cls_id] if cls_id < len(self.class_names) else 'Unknown'

            logger.debug(f"\n→ Detection {idx + 1}")
            logger.debug(f"Class ID: {cls_id}, Label: {label}")
            logger.debug(f"Confidence Score: {conf:.4f}")
            logger.debug(f"Raw Box Coords: x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")

            if conf < 0.3:
                logger.debug("Detection discarded due to low confidence.\n")
                continue

            # Reverse letterbox
            x1 = (x1 - pad[0]) / scale[0]
            y1 = (y1 - pad[1]) / scale[1]
            x2 = (x2 - pad[0]) / scale[0]
            y2 = (y2 - pad[1]) / scale[1]
            box = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

            logger.debug(f"Rescaled Box: x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}")

            detections.append({
                'box': box,
                'score': conf,
                'class_id': cls_id
            })

        logger.debug(f"\nTotal Valid Detections: {len(detections)}\n")
        return detections
