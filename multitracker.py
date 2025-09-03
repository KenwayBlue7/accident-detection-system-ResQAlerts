from track import Track
from utils import compute_overlap_ratio
from hungarian_matcher import HungarianMatcher
import numpy as np
import logging

logging.basicConfig(filename="output/logs/multitracker_log.txt", level=logging.DEBUG, filemode='w', format='[%(levelname)s] %(message)s')


class MultiTracker:
    def __init__(self, dist_threshold=150, max_skipped_frames=15, min_track_age=1, iou_threshold=0.0):
        self.tracks = []
        self.track_id_counter = 0
        self.dist_threshold = dist_threshold
        self.max_skipped_frames = max_skipped_frames
        self.min_track_age = min_track_age
        self.iou_threshold = iou_threshold
        self.matcher = HungarianMatcher(max_distance=self.dist_threshold)

    def update(self, detections):
        logging.debug("\n====================== [MULTITRACKER UPDATE START] ======================\n")
        logging.debug(f"[INFO] Updating with {len(detections)} detections")
        logging.debug(f"[INFO] Active Tracks: {len(self.tracks)}\n")

        # Step 1: Kalman predict for existing tracks
        for track in self.tracks:
            track.predict()

        # Step 2: No tracks yet → Create new tracks
        if len(self.tracks) == 0:
            logging.debug("[INFO] No existing tracks. Creating new tracks...\n")
            for i, det in enumerate(detections):
                det_center = ((det['box'][0] + det['box'][2]) // 2,
                              (det['box'][1] + det['box'][3]) // 2)
                new_track = Track(self.track_id_counter, det_center)
                new_track.box = det['box']
                new_track.class_id = det['class_id']
                self.tracks.append(new_track)
                logging.debug(f"  ➤ New Track {self.track_id_counter} created at {det_center}")
                self.track_id_counter += 1
            return

        # Step 3: Build cost matrix
        cost_matrix = []
        for t_idx, track in enumerate(self.tracks):
            row = []
            logging.debug(f"  ➤ Track {t_idx} Predicted Center: {track.prediction}")
            for d_idx, det in enumerate(detections):
                det_center = ((det['box'][0] + det['box'][2]) // 2,
                              (det['box'][1] + det['box'][3]) // 2)
                dist = ((track.prediction[0] - det_center[0]) ** 2 +
                        (track.prediction[1] - det_center[1]) ** 2) ** 0.5
                if dist > self.dist_threshold:
                    dist = 1e6  # Prevent assignment
                row.append(round(dist, 2))
                logging.debug(f"     → Detection {d_idx} Center: {det_center} | Distance: {dist:.2f}")
            cost_matrix.append(row)

        # Log cost matrix
        logging.debug("\n[INFO] Tabular Cost Matrix:")
        header = "       " + "".join([f"D{j:<8}" for j in range(len(detections))])
        logging.debug(header)
        for i, row in enumerate(cost_matrix):
            row_str = f"T{i:<5} " + "".join([f"{val:<9.2f}" for val in row])
            logging.debug(row_str)
        logging.debug("\n")

        # Step 4: Hungarian Matching
        matches = self.matcher.solve(np.array(cost_matrix))
        logging.debug(f"[INFO] Matching Result: {matches}\n")

        matched_tracks = set()
        matched_detections = set()

        # Step 5: Assign matches
        for t_idx, d_idx in enumerate(matches):
            if d_idx == -1 or d_idx >= len(detections):
                continue

            dist = cost_matrix[t_idx][d_idx]
            if dist >= self.dist_threshold or dist >= 1e6:
                logging.debug(f"[REJECTED] Match: Track {t_idx} ↔ Detection {d_idx} rejected due to distance ({dist:.2f})")
                continue

            if d_idx in matched_detections:
                logging.debug(f"[SKIPPED] Detection {d_idx} already matched. Skipping duplicate match.")
                continue

            det = detections[d_idx]
            iou = compute_overlap_ratio(self.tracks[t_idx].box, det['box']) if self.tracks[t_idx].box else 1.0
            if iou < self.iou_threshold:
                logging.debug(f"[REJECTED] Match: Track {t_idx} ↔ Detection {d_idx} rejected due to IOU {iou:.2f}")
                continue

            # Valid match → update track
            old_pos = self.tracks[t_idx].prediction
            det_center = ((det['box'][0] + det['box'][2]) // 2,
                          (det['box'][1] + det['box'][3]) // 2)

            # ✅ Pass class_id explicitly
            self.tracks[t_idx].update(det_center, detection_box=det['box'], class_id=det['class_id'])
            new_pos = self.tracks[t_idx].prediction

            dx = float(new_pos[0]) - float(old_pos[0])
            dy = float(new_pos[1]) - float(old_pos[1])
            logging.debug(f"[UPDATE] Track {t_idx} matched → Detection {d_idx} | Δx={dx:.2f}, Δy={dy:.2f}")

            self.tracks[t_idx].reset_missed()

            matched_tracks.add(t_idx)
            matched_detections.add(d_idx)

        # Step 6: Handle unmatched tracks
        for t_idx in range(len(self.tracks)):
            if t_idx not in matched_tracks:
                self.tracks[t_idx].mark_missed()
                logging.debug(f"[INFO] Track {t_idx} unmatched → Marked as missed")

        # Step 7: Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                det_center = ((det['box'][0] + det['box'][2]) // 2,
                              (det['box'][1] + det['box'][3]) // 2)
                new_track = Track(self.track_id_counter, det_center)
                new_track.box = det['box']
                new_track.class_id = det['class_id']
                self.tracks.append(new_track)
                logging.debug(f"[NEW] New Track {self.track_id_counter} created for unmatched Detection {d_idx} at {det_center}")
                self.track_id_counter += 1

        # Step 8: Cleanup old tracks
        before = len(self.tracks)
        self.tracks = [t for t in self.tracks if t.skipped_frames <= self.max_skipped_frames]
        after = len(self.tracks)
        if before != after:
            logging.debug(f"[CLEANUP] Removed {before - after} stale tracks")

        logging.debug(f"[INFO] Active Tracks After Update: {len(self.get_active_tracks())}")
        logging.debug("\n======================= [MULTITRACKER UPDATE END] =======================\n")

    def get_active_tracks(self):
        return [t for t in self.tracks if len(t.trace) >= self.min_track_age and t.box is not None]
