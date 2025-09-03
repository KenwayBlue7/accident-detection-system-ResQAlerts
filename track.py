from kalman_filter import KalmanTracker
import time
from collections import deque
import numpy as np
import math
import logging
import statistics

logging.basicConfig(filename="output/logs/track_log.txt", level=logging.DEBUG, filemode='w',
                    format='[%(levelname)s] %(message)s')
logger = logging.getLogger("TrackLogger")


class Track:
    def __init__(self, track_id, initial_position, dt=1.0, accel_noise_mag=0.5,
                 buffer_size=10, ftpp=30.0, speed_interval=5, update_freq=1):
        self.track_id = track_id
        self.class_id = None
        self.kalman = KalmanTracker(initial_position, dt, accel_noise_mag)
        self.prediction = initial_position
        self.trace = [initial_position]

        self.positions_buffer = deque(maxlen=buffer_size)
        self.timestamps_buffer = deque(maxlen=buffer_size)
        self.positions_buffer.append(initial_position)
        self.timestamps_buffer.append(time.time())

        self.velocity = 0.0
        self.speed_vector = (0.0, 0.0)
        self.ftpp = ftpp
        self.update_freq = update_freq
        self.speed_update_counter = 0
        self.prev_velocity = 0.0

        self.speed_history = deque(maxlen=10)
        self.smoothed_velocity = 0.0

        self.max_speed_seen = 0.0                     # ✅ NEW: Track maximum historical speed
        self.motion_threshold = 2.0                   # ✅ NEW: Minimum motion speed threshold (in mph)

        self.class_history = deque(maxlen=7)
        self.final_class_id = None
        self.class_change_confirm_counter = 0
        self.is_possible_class_switch = False

        self.skipped_frames = 0
        self.box = None
        self.outline = None
        self.future_trace = []

    def predict(self):
        if len(self.positions_buffer) >= 2:
            dx_est = (self.positions_buffer[-1][0] - self.positions_buffer[-2][0]) / (self.timestamps_buffer[-1] - self.timestamps_buffer[-2] + 1e-6)
            dy_est = (self.positions_buffer[-1][1] - self.positions_buffer[-2][1]) / (self.timestamps_buffer[-1] - self.timestamps_buffer[-2] + 1e-6)
            self.kalman.kalman.statePre[2] = dx_est
            self.kalman.kalman.statePre[3] = dy_est
        self.prediction = self.kalman.predict(debug=False)
        return self.prediction

    def update(self, detection_position, detection_box=None, class_id=None, data_correct=True):
        updated_position = self.kalman.update(detection_position, data_correct)
        self.trace.append(updated_position)
        if len(self.trace) > 30:
            self.trace = self.trace[-30:]
        self.prediction = updated_position

        timestamp_now = time.time()
        self.positions_buffer.append(updated_position)
        self.timestamps_buffer.append(timestamp_now)

        if detection_box:
            self.box = detection_box

        if class_id is not None:
            self.class_history.append(class_id)
            try:
                most_common_class = statistics.median(self.class_history)
            except statistics.StatisticsError:
                most_common_class = class_id

            if self.final_class_id is not None and most_common_class != self.final_class_id:
                self.class_change_confirm_counter += 1
                if self.class_change_confirm_counter >= 3:
                    self.is_possible_class_switch = True
            else:
                self.class_change_confirm_counter = 0
                self.is_possible_class_switch = False

            self.final_class_id = most_common_class
            self.class_id = self.final_class_id

        self.compute_velocity()

    def compute_velocity(self):
        state_post = self.kalman.kalman.statePost
        dx = float(state_post[2])
        dy = float(state_post[3])
        raw_speed_mph = math.sqrt(dx ** 2 + dy ** 2) * self.ftpp * 3600 / 5280

        # ✅ Apply minimum motion threshold filter
        if raw_speed_mph < self.motion_threshold:
            filtered_speed = 0.0
        else:
            filtered_speed = raw_speed_mph

        if filtered_speed > 0.01:
            self.velocity = filtered_speed
            self.speed_vector = (round(dx, 2), round(dy, 2))

        self.speed_history.append(self.velocity)

        if len(self.speed_history) > 0:
            self.smoothed_velocity = statistics.median(self.speed_history)
        else:
            self.smoothed_velocity = self.velocity

        # ✅ Update maximum speed seen
        self.max_speed_seen = max(self.max_speed_seen, self.smoothed_velocity)

    def mark_missed(self):
        self.skipped_frames += 1

    def reset_missed(self):
        self.skipped_frames = 0
