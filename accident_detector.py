import math
from collections import defaultdict, deque
from utils import compute_overlap_ratio
import logging
import os
import statistics

# Setup logging
os.makedirs("output/logs", exist_ok=True)
logging.basicConfig(filename="output/logs/accident_log.txt", level=logging.DEBUG, filemode='w',
                    format='[%(levelname)s] %(message)s')
logger = logging.getLogger("AccidentDetectorLogger")

class AccidentDetector:
    def __init__(self,
                 speed_threshold=6.0,
                 overlap_threshold=0.3,
                 relative_speed_threshold=6.0,
                 center_proximity_threshold=30,
                 score_threshold=3,
                 confirm_frames=5,
                 accident_display_duration=40,
                 speed_drop_confirm_frames=5,
                 speed_drop_threshold=70,
                 possible_accident_threshold=50,
                 class_freeze_frames=10,
                 min_required_max_speed=20.0):  # Threshold to avoid parked object misdetections

        self.speed_threshold = speed_threshold
        self.overlap_threshold = overlap_threshold
        self.relative_speed_threshold = relative_speed_threshold
        self.center_proximity_threshold = center_proximity_threshold
        self.score_threshold = score_threshold
        self.confirm_frames = confirm_frames
        self.accident_display_duration = accident_display_duration
        self.speed_drop_confirm_frames = speed_drop_confirm_frames
        self.speed_drop_threshold = speed_drop_threshold
        self.possible_accident_threshold = possible_accident_threshold
        self.class_freeze_frames = class_freeze_frames
        self.min_required_max_speed = min_required_max_speed  # ✅ NEW

        self.confirmation_counter = defaultdict(int)
        self.accident_display_counter = {}
        self.recent_accidents = set()
        self.possible_accidents = set()

        self.last_velocities = {}
        self.low_speed_frame_count = defaultdict(int)
        self.class_history = defaultdict(lambda: deque(maxlen=class_freeze_frames))
        self.frozen_class = {}
        self.class_change_counter = defaultdict(int)

    def detect_accidents(self, tracks):
        accidents = []
        if len(tracks) < 2:
            return []

        for track in tracks:
            if track.track_id not in self.last_velocities:
                self.last_velocities[track.track_id] = track.smoothed_velocity

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                t1, t2 = tracks[i], tracks[j]
                if t1.box is None or t2.box is None:
                    continue

                # Skip if both tracks were always slow (max speed never crossed threshold)
                if t1.max_speed_seen < self.min_required_max_speed and t2.max_speed_seen < self.min_required_max_speed:
                    continue

                pair_id = tuple(sorted((t1.track_id, t2.track_id)))

                # Overlap and center proximity
                overlap = compute_overlap_ratio(t1.box, t2.box)
                cx1 = t1.box[0] + t1.box[2] // 2
                cy1 = t1.box[1] + t1.box[3] // 2
                cx2 = t2.box[0] + t2.box[2] // 2
                cy2 = t2.box[1] + t2.box[3] // 2
                center_dist = math.hypot(cx1 - cx2, cy1 - cy2)

                # Relative speed and low speed
                rel_speed = math.hypot(t1.speed_vector[0] - t2.speed_vector[0],
                                       t1.speed_vector[1] - t2.speed_vector[1])
                low_speed = t1.smoothed_velocity < self.speed_threshold and t2.smoothed_velocity < self.speed_threshold

                # Sudden stop detection
                prev_v1 = self.last_velocities.get(t1.track_id, 0)
                sudden_stop1 = prev_v1 >= self.speed_drop_threshold and t1.smoothed_velocity < self.speed_threshold
                possible_stop1 = self.possible_accident_threshold <= prev_v1 < self.speed_drop_threshold and t1.smoothed_velocity < self.speed_threshold

                if sudden_stop1 or possible_stop1:
                    self.low_speed_frame_count[t1.track_id] += 1
                else:
                    self.low_speed_frame_count[t1.track_id] = 0

                persistent_stop1 = self.low_speed_frame_count[t1.track_id] >= self.speed_drop_confirm_frames

                # Class history and stabilization
                self.class_history[t1.track_id].append(t1.class_id)
                self.class_history[t2.track_id].append(t2.class_id)

                try:
                    median_class_t1 = int(statistics.median(self.class_history[t1.track_id]))
                except:
                    median_class_t1 = t1.class_id
                try:
                    median_class_t2 = int(statistics.median(self.class_history[t2.track_id]))
                except:
                    median_class_t2 = t2.class_id

                # Class freezing
                if t1.track_id not in self.frozen_class:
                    self.frozen_class[t1.track_id] = median_class_t1
                if t2.track_id not in self.frozen_class:
                    self.frozen_class[t2.track_id] = median_class_t2

                # Class switch detection
                if median_class_t1 != self.frozen_class[t1.track_id]:
                    self.class_change_counter[t1.track_id] += 1
                    t1.is_possible_class_switch = True
                    self.possible_accidents.add(t1.track_id)
                else:
                    self.class_change_counter[t1.track_id] = 0
                    t1.is_possible_class_switch = False

                if median_class_t2 != self.frozen_class[t2.track_id]:
                    self.class_change_counter[t2.track_id] += 1
                    t2.is_possible_class_switch = True
                    self.possible_accidents.add(t2.track_id)
                else:
                    self.class_change_counter[t2.track_id] = 0
                    t2.is_possible_class_switch = False

                # Skip collision if both are slow (speed < 20)
                '''
                if t1.smoothed_velocity < 8 and t2.smoothed_velocity < 8:
                    logger.debug(f"[SKIP] Skipping pair {pair_id} — Both speeds below 20 mph (t1: {t1.smoothed_velocity:.2f}, t2: {t2.smoothed_velocity:.2f})")
                    continue

                # Final accident scoring
                score = 0
                if overlap >= self.overlap_threshold:
                    score += 1
                if center_dist <= self.center_proximity_threshold:
                    score += 1
                if low_speed:
                    score += 1
                if rel_speed < self.relative_speed_threshold:
                    score += 1
                if sudden_stop1 and persistent_stop1:
                    score += 1
                elif possible_stop1 and persistent_stop1:
                    self.possible_accidents.add(t1.track_id)
                '''

                if t1.smoothed_velocity < 8 and t2.smoothed_velocity < 8:
                    low_speed_pair = True
                else:
                    low_speed_pair = False

                score = 0
                if overlap >= self.overlap_threshold:
                    score += 1
                    if overlap >= (self.overlap_threshold + 0.2):
                        score += 1  # bonus point for strong overlap
                if center_dist <= self.center_proximity_threshold:
                    score += 1
                if low_speed and not low_speed_pair:  # only count if not both are slow
                    score += 1
                if rel_speed < self.relative_speed_threshold:
                    score += 1
                if sudden_stop1 and persistent_stop1:
                    score += 1
                elif possible_stop1 and persistent_stop1:
                    self.possible_accidents.add(t1.track_id)

                # Confirm or discard
                if score >= self.score_threshold:
                    self.confirmation_counter[pair_id] += 1
                else:
                    self.confirmation_counter[pair_id] = max(0, self.confirmation_counter[pair_id] - 1)

                if self.confirmation_counter[pair_id] >= self.confirm_frames:
                    if pair_id not in self.recent_accidents:
                        self.recent_accidents.add(pair_id)
                        self.accident_display_counter[pair_id] = self.accident_display_duration
                    accidents.append(pair_id)

        # Display timeout management
        expired = []
        for pair in list(self.accident_display_counter.keys()):
            self.accident_display_counter[pair] -= 1
            if self.accident_display_counter[pair] <= 0:
                expired.append(pair)

        for pair in expired:
            self.recent_accidents.discard(pair)
            self.accident_display_counter.pop(pair, None)
            self.confirmation_counter[pair] = 0

        # Update last speeds
        for track in tracks:
            self.last_velocities[track.track_id] = track.smoothed_velocity

        return list(self.recent_accidents)
