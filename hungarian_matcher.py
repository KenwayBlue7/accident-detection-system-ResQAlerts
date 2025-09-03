import numpy as np
import logging
from scipy.optimize import linear_sum_assignment

# Setup logging
logging.basicConfig(filename="output/logs/hungarian_log.txt", level=logging.DEBUG, filemode='w', format='[%(levelname)s] %(message)s')
logger = logging.getLogger("HungarianMatcherLogger")

class HungarianMatcher:
    def __init__(self, max_distance=100.0):
        """
        :param max_distance: Maximum acceptable matching cost (distance threshold).
        """
        self.max_distance = max_distance

    def solve(self, cost_matrix):
        """
        Perform optimal assignment using the Hungarian Algorithm.

        :param cost_matrix: 2D numpy array of shape (num_tracks, num_detections)
        :return: assignment list where index is track index, value is detection index (or -1 if unmatched)
        """
        logger.debug("\n=========== [HUNGARIAN MATCHER START] ===========")
        if cost_matrix.size == 0:
            logger.debug("Empty cost matrix received. Returning empty result.")
            return []

        num_tracks, num_detections = cost_matrix.shape
        logger.debug(f"Cost Matrix Shape: {cost_matrix.shape}")
        logger.debug(f"Raw Cost Matrix:\n{np.round(cost_matrix, 2)}")

        # Run Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignment = [-1] * num_tracks
        for i in range(len(row_ind)):
            track_idx = row_ind[i]
            det_idx = col_ind[i]
            cost = cost_matrix[track_idx][det_idx]

            if cost <= self.max_distance:
                assignment[track_idx] = det_idx
                logger.debug(f"✅ Track {track_idx} matched to Detection {det_idx} | Cost: {cost:.2f}")
            else:
                logger.debug(f"❌ Match Rejected: Track {track_idx} → Detection {det_idx} | Cost: {cost:.2f} > max_distance ({self.max_distance})")

        # Log unmatched tracks
        for i, val in enumerate(assignment):
            if val == -1:
                logger.debug(f"⚠ Track {i} could not be matched (unassigned)")

        logger.debug(f"Final Assignment Result: {assignment}")
        logger.debug("=========== [SCIPY HUNGARIAN MATCHER END] ===========\n")
        return assignment
