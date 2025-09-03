import cv2
import time
import os
from detector import YOLODetector
from multitracker import MultiTracker
from utils import draw_detections, letterbox_resize
from accident_detector import AccidentDetector
import logging
from firebase_service import upload_accident_data, send_accident_notification

# Setup Logging
os.makedirs("output/logs", exist_ok=True)
logging.basicConfig(filename="output/logs/main_log.txt", level=logging.DEBUG, filemode='w', format='[%(levelname)s] %(message)s')
logger = logging.getLogger("MainLogger")

def main():
    video_path = "videos/accident4.mp4"
    model_path = "models/yolov7_5classes_640x640.onnx"
    class_names_path = "data/traffic.names"
    output_path = "output/accident_output.avi"
    confidence_threshold = 0.3

    # Hardcoded coordinates for each video
    locations = [
        (13.0827, 80.2707),
        (10.903356, 76.434035),
        (9.9252, 78.1198),
        (10.927301372158329, 76.46369952927397),
        (8.5241, 76.9366)
    ]
    selected_coordinates = locations[3]  # Change this index per video

    logger.debug("\n========== [MAIN MODULE START] ==========")
    logger.debug(f"Video path: {video_path}")
    logger.debug(f"Model path: {model_path}")
    logger.debug(f"Class names path: {class_names_path}")
    logger.debug(f"Output path: {output_path}")
    logger.debug(f"Confidence threshold: {confidence_threshold}\n")

    detector = YOLODetector(model_path, class_names_path, use_gpu=True, input_size=(640, 640))
    tracker = MultiTracker(dist_threshold=60, max_skipped_frames=30, min_track_age=1)
    accident_detector = AccidentDetector(
        speed_threshold=6.0,
        overlap_threshold=0.2,
        relative_speed_threshold=8.0,
        center_proximity_threshold=50,
        confirm_frames=5,
        accident_display_duration=40
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"[ERROR] Could not open video: {video_path}")
        print(f"[ERROR] Could not open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.debug(f"[VIDEO INFO] Resolution: {width}x{height}, FPS: {fps:.2f}\n")
    print(f"[INFO] Video resolution: {width}x{height}, FPS: {fps:.2f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0
    tracking_log_path = "output/tracking_log.txt"
    log_file = open(tracking_log_path, "w")
    accident_event_start_time = None
    latest_snapshot_path = None
    accident_upload_completed = False
    last_confirmed_frame = None
    last_confirmed_path = None
    last_detection_time = None
    POST_ACCIDENT_WAIT = 7  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.debug("\n[END] End of video stream reached.")
            break

        logger.debug(f"\n========== [FRAME {frame_index}] ==========")
        letterbox_frame, scale, pad = letterbox_resize(frame, target_size=(640, 640))
        detections = detector.detect(letterbox_frame)

        detections = [det for det in detections if det['score'] >= confidence_threshold]
        for i, det in enumerate(detections):
            x, y, w, h = det['box']
            x = int((x - pad[0]) / scale[0])
            y = int((y - pad[1]) / scale[1])
            w = int(w / scale[0])
            h = int(h / scale[1])
            det['box'] = (x, y, w, h)

        tracker.update(detections)
        active_tracks = tracker.get_active_tracks()

        log_file.write(f"Frame {frame_index}:\n")
        for track in active_tracks:
            class_id = int(track.class_id) if track.class_id is not None else -1
            label = detector.class_names[class_id] if 0 <= class_id < len(detector.class_names) else "unknown"
            log_file.write(f"  Track ID {track.track_id} ({label}) - Box: {track.box}\n")

        accidents = accident_detector.detect_accidents(active_tracks)
        logger.debug(f"[ACCIDENT DETECTOR] Accidents detected: {len(accidents)}")

        current_time = time.time()

        if accidents:
            if accident_event_start_time is None:
                accident_event_start_time = current_time
                latest_snapshot_path = None
                accident_upload_completed = False

            latest_snapshot_path = f"output/snapshot_frame_{frame_index}.jpg"
            cv2.imwrite(latest_snapshot_path, frame)

            # Save latest confirmed accident frame
            last_confirmed_frame = frame.copy()
            last_confirmed_path = f"output/last_confirmed_frame.jpg"
            cv2.imwrite(last_confirmed_path, last_confirmed_frame)
            last_detection_time = current_time

            for tid1, tid2 in accidents:
                for track in active_tracks:
                    if track.track_id in (tid1, tid2) and track.box:
                        x, y, w, h = track.box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        label = "ACCIDENT"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 2
                        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
                        text_x = x
                        text_y = y + h + text_size[1] + 10
                        cv2.rectangle(frame, (text_x - 4, text_y - text_size[1] - 4),
                                      (text_x + text_size[0] + 4, text_y + 4), (0, 0, 0), -1)
                        cv2.putText(frame, label, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

            # Upload after 10 seconds of detection
            if (current_time - accident_event_start_time >= 10 and latest_snapshot_path and not accident_upload_completed):
                logger.info("[UPLOAD] Uploading snapshot after 10s of detection.")
                image_url, maps_link = upload_accident_data(latest_snapshot_path, selected_coordinates)
                send_accident_notification("⚠ Accident Detected", "Accident detected near your location.", image_url, maps_link)
                accident_upload_completed = True

        else:
            accident_event_start_time = None
            latest_snapshot_path = None
            accident_upload_completed = False

        # Possible Accidents
        for track in active_tracks:
            if track.track_id in accident_detector.possible_accidents:
                x, y, w, h = track.box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(frame, "POSSIBLE ACCIDENT", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Class Switch
        for track in active_tracks:
            if track.is_possible_class_switch:
                x, y, w, h = track.box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(frame, "POSSIBLE CLASS SWITCH", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        draw_detections(frame, active_tracks, detector.class_names)
        out_writer.write(frame)
        cv2.imshow("Detection & Tracking", frame)

        if cv2.waitKey(1) == ord('q'):
            logger.debug("[USER ACTION] Quit triggered via 'q' key.")
            break

        frame_index += 1

    cap.release()
    out_writer.release()
    log_file.close()

    # Post-processing wait and upload
    if last_confirmed_frame is not None:
        logger.info("[POST PROCESS] Waiting for 7 seconds for final upload check...")
        start_wait = time.time()
        while time.time() - start_wait < POST_ACCIDENT_WAIT:
            time.sleep(1)

        logger.info("[POST PROCESS] Uploading final confirmed snapshot...")
        image_url, maps_link = upload_accident_data(last_confirmed_path, selected_coordinates)
        send_accident_notification("⚠ Accident Detected (Final)", "Final confirmed snapshot uploaded.", image_url, maps_link)

    logger.info(f"[OUTPUT] Tracking log saved at: {tracking_log_path}")
    logger.info(f"[OUTPUT] Output video saved to: {output_path}")
    print(f"[INFO] Tracking log saved at: {tracking_log_path}")
    print(f"[INFO] Output video saved to: {output_path}")

if __name__ == "__main__":
    main()
