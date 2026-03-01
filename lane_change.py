"""
Lane-change detection from a sequence of predicted steering angles.

Two complementary heuristics:
    1. Zero-crossing: the angle changes sign and exceeds a magnitude threshold.
    2. Sustained derivative: the mean |d(angle)/dt| stays above a threshold
       for several consecutive frames.

Both thresholds are configured in config.py.
"""
import logging
from typing import Optional, Tuple
import numpy as np

from config import (
    LANE_CHANGE_WINDOW,
    LANE_CHANGE_ANGLE_THRESHOLD,
    LANE_CHANGE_DERIVATIVE_THRESHOLD,
    LANE_CHANGE_SUSTAINED_FRAMES,
)


class LaneChangeDetector:
    """
    Stateful detector. Feed predicted steering angles one at a time via
    `update(angle)`. The detector maintains an internal sliding window and
    returns a detection result after each update.
    """

    def __init__(
        self,
        window: int = LANE_CHANGE_WINDOW,
        angle_threshold: float = LANE_CHANGE_ANGLE_THRESHOLD,
        deriv_threshold: float = LANE_CHANGE_DERIVATIVE_THRESHOLD,
        sustained: int = LANE_CHANGE_SUSTAINED_FRAMES
    ):
        self.window = window
        self.angle_threshold = angle_threshold
        self.deriv_threshold = deriv_threshold
        self.sustained = sustained
        self._history: list[float] = []

    def update(self, angle: float) -> Tuple[bool, Optional[str]]:
        """Update the detector with a new steering angle.

        Parameters
        ----------
        angle : float
            The latest predicted steering angle (in range [-1, 1]).

        Returns
        -------
        Tuple[bool, Optional[str]]
            A tuple where the first element is a boolean indicating whether a
            lane change was detected, and the second element is the direction
            ("LEFT" or "RIGHT") if a change was detected, or None otherwise.
        """
        self._history.append(angle)

        # Keep only the last `window` entries
        if len(self._history) > self.window:
            self._history = self._history[-self.window:]

        if len(self._history) < 3:
            return False, None

        arr = np.array(self._history)

        # Heuristic 1: zero-crossing + magnitude threshold
        signs = np.sign(arr)
        crossings = np.where(np.diff(signs) != 0)[0]
        if len(crossings) > 0:
            # Check if the absolute angle exceeded the threshold at any point
            if np.max(np.abs(arr)) >= self.angle_threshold:
                direction = "RIGHT" if arr[-1] > 0 else "LEFT"
                logging.info(f"Lane change detected: {direction} (angle crossed zero with max |angle|={np.max(np.abs(arr)):.3f})")
                return True, direction

        # Heuristic 2: sustained large derivative
        derivative = np.abs(np.diff(arr))
        if len(derivative) >= self.sustained:
            # Sliding check: any run of `sustained` consecutive frames above threshold
            for start in range(len(derivative) - self.sustained + 1):
                window_deriv = derivative[start: start + self.sustained]
                if np.all(window_deriv >= self.deriv_threshold):
                    direction = "RIGHT" if arr[-1] > arr[start] else "LEFT"
                    logging.info(f"Lane change detected: {direction} (sustained derivative with max |d(angle)/dt|={np.max(window_deriv):.3f})")
                    return True, direction

        return False, None

    def reset(self):
        """Clear internal history."""
        self._history.clear()


def detect_lane_changes(angles: np.ndarray) -> list[Tuple[int, str]]:
    """Detect lane changes in a sequence of predicted steering angles.

    Parameters
    ----------
    angles : np.ndarray
        Sequence of predicted steering angles

    Returns
    -------
    list[Tuple[int, str]]
        A list of detected lane change events, where each event is a tuple
        containing the index of the frame where the change was detected and
        the direction ("LEFT" or "RIGHT").
    """
    detector = LaneChangeDetector()
    results = []
    for i, angle in enumerate(angles):
        found, direction = detector.update(float(angle))
        if found:
            results.append((i, direction))
            logging.debug(f"Lane change event at index {i}: {direction}")
            detector.reset()  # avoid re-triggering on the same event
    return results
