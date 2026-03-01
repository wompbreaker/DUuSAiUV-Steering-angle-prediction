"""
Inference / prediction script.

Loads a trained model, runs it on test images, draws steering-angle
overlays, detects lane changes, and produces annotated output frames
(individual images and / or a video).

Usage:
    python predict.py                        # defaults
    python predict.py --video                # also produce .mp4
    python predict.py --max_frames 200       # limit number of frames
    python predict.py --model saved_model/best_model.keras
"""

import argparse
import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import cv2
import keras

# Set config variables before importing other modules that rely on them
import config
from data_loader import preprocess_image, fix_path
from lane_change import LaneChangeDetector
from utils import (
    draw_steering_overlay,
    draw_lane_change_warning,
    make_video_writer,
)


def _load_raw_image(path: str) -> np.ndarray:
    """Load image as-is (BGR uint8) for display purposes."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def run_inference(
    model_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    save_video: bool = False,
    show_live: bool = False,
    max_samples: Optional[int] = None,
):
    """Run inference on test images, draw overlays, and save annotated frames.

    The function processes a sequence of images, maintaining a buffer of the last SEQ_LEN frames
    for temporal inference. It draws the predicted steering angle as an overlay on each frame,
    compares it to the ground truth, and detects lane changes based on the predicted angles.

    Parameters
    ----------
    model_path : Optional[str], optional
        Path to the saved Keras model. Defaults to best_model.keras.
    max_frames : Optional[int], optional
        Maximum number of frames to process. None = all. Defaults to None.
    save_video : bool, optional
        If True, write an MP4 video to outputs/. Defaults to False.
    show_live : bool, optional
        If True, display frames with cv2.imshow (press 'q' to quit). Defaults to False.
    max_samples : Optional[int], optional
        Limit CSV rows read (for quick testing). Defaults to None.
    """

    # --- Load model --- #
    if model_path is None:
        model_path = os.path.join(config.SAVED_MODEL_DIR, "best_model.keras")
    logging.info(f"Loading model from {model_path}...")
    model: keras.Model = keras.models.load_model(model_path)
    model.summary()

    # --- Load driving log --- #
    columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
    dataframe = pd.read_csv(config.CSV_PATH, names=columns, skiprows=1)
    if max_samples is not None:
        dataframe = dataframe.iloc[:max_samples]

    # Use only centre camera for inference visualisation
    image_paths = [fix_path(p) for p in dataframe["center"].values]
    true_steerings = dataframe["steering"].astype(float).values
    total = len(image_paths) if max_frames is None else min(max_frames, len(image_paths))

    # --- Prepare output dir --- #
    frames_dir = os.path.join(config.OUTPUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # --- Warm-up: fill the initial sequence buffer --- #
    seq_buffer: list[np.ndarray] = []  # preprocessed frames

    video_writer = None
    detector = LaneChangeDetector()
    predicted_angles: list[float] = []

    logging.info(f"Running inference on {total} frames ...")

    for idx in range(total):
        raw = _load_raw_image(image_paths[idx])
        preprocessed = preprocess_image(raw)
        seq_buffer.append(preprocessed)

        if len(seq_buffer) < config.SEQ_LEN:
            # Not enough frames yet - just show the raw image with ground truth
            display = raw.copy()
            cv2.putText(
                display,
                f"Buffering {len(seq_buffer)}/{config.SEQ_LEN}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
            annotated = display
        else:
            # Keep only last SEQ_LEN frames
            if len(seq_buffer) > config.SEQ_LEN:
                seq_buffer = seq_buffer[-config.SEQ_LEN:]

            # Build input tensor: (1, SEQ_LEN, H, W, C)
            x = np.array(seq_buffer)[np.newaxis, ...]
            pred = model.predict(x, verbose=0)[0, 0]
            pred = float(np.clip(pred, -1.0, 1.0))
            predicted_angles.append(pred)

            # Resize raw frame for nice display (optional, keep original size)
            display = raw.copy()

            # Draw steering overlay
            annotated = draw_steering_overlay(display, pred)

            # Draw ground truth for comparison (degrees)
            gt = true_steerings[idx]
            gt_deg = gt * 90.0
            gt_text = f"GT: {gt_deg:+.1f} deg"
            cv2.putText(
                annotated, gt_text, (8, annotated.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA,
            )

            # Lane change detection
            detected, direction = detector.update(pred)
            if detected and direction is not None:
                annotated = draw_lane_change_warning(annotated, direction)
                logging.info(f"  [LANE CHANGE] frame {idx}: {direction}")

        # --- Save / display frame --- #
        frame_path = os.path.join(frames_dir, f"frame_{idx:05d}.jpg")
        cv2.imwrite(frame_path, annotated)

        if save_video:
            if video_writer is None:
                h, w = annotated.shape[:2]
                vid_path = os.path.join(config.OUTPUT_DIR, "prediction.mp4")
                video_writer = make_video_writer(vid_path, fps=20.0, width=w, height=h)
            video_writer.write(annotated)

        if show_live:
            cv2.imshow("Steering Prediction", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.info("User quit.")
                break

        if (idx + 1) % 100 == 0:
            logging.info(f"  Processed {idx + 1}/{total} frames")

    # --- Cleanup --- #
    if video_writer is not None:
        video_writer.release()
        logging.info(f"Video saved to {os.path.join(config.OUTPUT_DIR, 'prediction.mp4')}")
    if show_live:
        cv2.destroyAllWindows()

    # --- Summary --- #
    if predicted_angles:
        pa = np.array(predicted_angles)
        logging.info(f"\nInference complete.")
        logging.info(f"  Frames processed   : {total}")
        logging.info(f"  Predictions made   : {len(pa)}")
        logging.info(f"  Predicted range    : [{pa.min():.3f}, {pa.max():.3f}]")
        logging.info(f"  Mean |pred - gt|   : "
              f"{np.mean(np.abs(pa - true_steerings[config.SEQ_LEN - 1: config.SEQ_LEN - 1 + len(pa)])):.4f}")
    logging.info(f"  Annotated frames in: {frames_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run steering angle inference")
    parser.add_argument(
        "-m", "--model", type=str, default=None,
        help="Path to saved .keras model (default: best_model.keras)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Max frames to process (default: all)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit CSV rows (for quick testing)"
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Produce an MP4 video of annotated frames"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Show frames live in a window (press 'q' to quit)"
    )
    return parser.parse_args()

# --- CLI --- #

if __name__ == "__main__":
    args = parse_args()
    run_inference(
        model_path=args.model,
        max_frames=args.max_frames,
        save_video=args.video,
        show_live=args.live,
        max_samples=args.max_samples,
    )