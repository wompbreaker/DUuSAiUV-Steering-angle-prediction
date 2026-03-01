import math
from typing import Tuple

import cv2
import numpy as np

from config import (
    VIS_STRAIGHT_THRESH,
    VIS_MODERATE_THRESH,
)


def _angle_color(angle: float) -> Tuple[int, int, int]:
    """Return a BGR colour based on the absolute steering angle."""
    a = abs(angle)
    if a < VIS_STRAIGHT_THRESH:
        return (0, 200, 0)  # green
    elif a < VIS_MODERATE_THRESH:
        return (0, 220, 255)  # yellow (BGR)
    else:
        return (0, 0, 255)  # red


def _draw_steering_wheel(
    img: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    angle_deg: float,
    colour: Tuple[int, int, int]
) -> None:
    """Draw a simple steering wheel with three spokes rotated by angle_deg.

    Parameters
    ----------
    img : np.ndarray
        Image to draw on (modified in-place)
    cx : int
        X-coordinate of the center of the steering wheel
    cy : int
        Y-coordinate of the center of the steering wheel
    radius : int
        Radius of the steering wheel
    angle_deg : float
        Steering angle in degrees (rotation of the wheel)
    colour : Tuple[int, int, int]
        BGR colour for the wheel
    """
    # Outer ring
    cv2.circle(img, (cx, cy), radius, colour, thickness=2, lineType=cv2.LINE_AA)
    # Inner hub
    hub_r = max(radius // 6, 3)
    cv2.circle(img, (cx, cy), hub_r, colour, thickness=-1, lineType=cv2.LINE_AA)

    # Spokes on the wheel at 120 degrees apart, rotated by angle_deg
    rot_rad = math.radians(angle_deg)
    spoke_len = radius - 2
    for base_deg in (0, 120, 240):
        theta = math.radians(base_deg) + rot_rad
        sx = int(cx + spoke_len * math.sin(theta))
        sy = int(cy - spoke_len * math.cos(theta))
        cv2.line(img, (cx, cy), (sx, sy), colour, thickness=2, lineType=cv2.LINE_AA)


def draw_steering_overlay(frame: np.ndarray, angle: float) -> np.ndarray:
    """Draw a steering wheel overlay and angle text on the input frame.

    Parameters
    ----------
    frame : np.ndarray
        Input image (H, W, 3) uint8 BGR
    angle : float
        Steering angle in range [-1, 1], where -1 = full left, +1 = full right

    Returns
    -------
    np.ndarray
        Annotated image (copy of input)
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Map [-1, 1] to degrees [-90°, 90°]
    angle_deg = angle * 90.0
    colour = _angle_color(angle)

    # Steering wheel: bottom-right corner area
    radius = min(h, w) // 8
    cx = w - radius - 15
    cy = h - radius - 15

    # Semi-transparent dark background disc for contrast
    overlay = out.copy()
    cv2.circle(overlay, (cx, cy), radius + 10, (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)

    _draw_steering_wheel(out, cx, cy, radius, angle_deg, colour)

    # Degree text
    text = f"{angle_deg:+.1f}°"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx, ty = 8, 20
    cv2.rectangle(out, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), (0, 0, 0), -1)
    cv2.putText(out, text, (tx, ty), font, font_scale, colour, thickness, cv2.LINE_AA)

    return out


def draw_lane_change_warning(frame: np.ndarray, direction: str) -> np.ndarray:
    """Draw a red border and warning text on the frame to indicate a detected lane change.

    Parameters
    ----------
    frame : np.ndarray
        Input image (H, W, 3) uint8 BGR
    direction : str
        Direction of the lane change ("LEFT" or "RIGHT")

    Returns
    -------
    np.ndarray
        Annotated image (copy of input)
    """
    out = frame.copy()
    h, w = out.shape[:2]
    border = 6

    # Red border
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), border)

    # Warning text
    text = f"LANE CHANGE: {direction}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = w - tw - 8
    ty = 20

    # Black background for contrast
    cv2.rectangle(out, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6), (0, 0, 0), -1)
    cv2.putText(out, text, (tx, ty), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    return out


def make_video_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create and return a cv2.VideoWriter for MP4 output.

    Parameters
    ----------
    path : str
        Output video file path (e.g., "output/prediction.mp4")
    fps : float
        Frames per second for the output video
    width : int
        Width of the output video in pixels
    height : int
        Height of the output video in pixels

    Returns
    -------
    cv2.VideoWriter
        OpenCV VideoWriter object for writing video frames
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))
