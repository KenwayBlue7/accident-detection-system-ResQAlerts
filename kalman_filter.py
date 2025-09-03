import cv2
import numpy as np
import logging

# Setup Kalman filter logging
logging.basicConfig(filename="output/logs/kalman_log.txt", level=logging.DEBUG, filemode='w', format='[%(levelname)s] %(message)s')
logger = logging.getLogger("KalmanLogger")

class KalmanTracker:
    def __init__(self, pt, dt=1.0, accel_noise_mag=0.5):
        self.kalman = cv2.KalmanFilter(4, 2)

        # Transition Matrix A
        self.kalman.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        # Measurement Matrix H
        self.kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)

        # Process noise covariance Q
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * accel_noise_mag

        # Measurement noise covariance R
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

        # Posteriori error estimate covariance matrix P
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

        # Initial state (x, y, dx, dy)
        self.kalman.statePost = np.array([
            [pt[0]],
            [pt[1]],
            [0.0],
            [0.0]
        ], dtype=np.float32)

        logger.debug("\n================= [INIT-KALMAN] =================")
        logger.debug(f"Initial Position: {pt}")
        logger.debug("Transition Matrix A:\n%s", np.round(self.kalman.transitionMatrix, 2))
        logger.debug("Measurement Matrix H:\n%s", np.round(self.kalman.measurementMatrix, 2))
        logger.debug("Process Noise Covariance Q:\n%s", np.round(self.kalman.processNoiseCov, 2))
        logger.debug("Measurement Noise Covariance R:\n%s", np.round(self.kalman.measurementNoiseCov, 2))
        logger.debug("Initial StatePost (x, y, dx, dy):\n%s", np.round(self.kalman.statePost, 2))
        logger.debug("==================================================\n")

    def debug_predict(self):
        logger.debug("\n================= [DEBUG-KALMAN] BEFORE predict() =================")
        logger.debug("StatePost (last corrected state):\n%s", np.round(self.kalman.statePost, 2))
        logger.debug("Transition Matrix (A):\n%s", np.round(self.kalman.transitionMatrix, 2))

        simulated_pred = np.dot(self.kalman.transitionMatrix, self.kalman.statePost)
        logger.debug("\nSimulated Predicted State (A * statePost):\n%s", np.round(simulated_pred, 2))

        actual_pred = self.kalman.predict()
        logger.debug("\nActual Predicted State from cv2.kalman.predict():\n%s", np.round(actual_pred, 2))

        logger.debug("\nComparison: Simulated vs Actual (should match closely)")
        logger.debug("Are they equal? %s", np.allclose(simulated_pred, actual_pred, atol=1e-4))
        logger.debug("====================================================================\n")

    def predict(self, debug=False):
        if debug:
            self.debug_predict()
        else:
            logger.debug("\n[PREDICT] Kalman Predict called (no debug mode)")

        pred = self.kalman.predict()
        logger.debug(f"[PREDICT] Predicted Output Position: ({pred[0][0]:.2f}, {pred[1][0]:.2f})\n")
        return (int(pred[0].item()), int(pred[1].item()))

    def update(self, pt, data_correct=True):
        logger.debug("\n================= [UPDATE-KALMAN] =================")
        logger.debug(f"Measurement Received: {pt}")
        measurement = np.array([[np.float32(pt[0])], [np.float32(pt[1])]])
        logger.debug(f"Measurement Vector (z):\n{np.round(measurement, 2)}")

        corrected = self.kalman.correct(measurement)
        logger.debug(f"Corrected StatePost (x, y, dx, dy):\n{np.round(self.kalman.statePost, 2)}")
        logger.debug(f"Corrected Output Position: ({corrected[0][0]:.2f}, {corrected[1][0]:.2f})")
        logger.debug("===================================================\n")

        return (int(corrected[0].item()), int(corrected[1].item()))
