# Predicting Steering Angle in a Self Driving Car

A CNN + LSTM deep learning pipeline that predicts steering angles from sequences of dashcam frames. The model processes temporal sequences of images through a convolutional feature extractor followed by an LSTM, outputting a continuous steering angle in the range [-1, 1]. The project also includes a heuristic-based lane-change detector that analyses the predicted steering signal in real time.

---

## Table of Contents

- [Predicting Steering Angle in a Self Driving Car](#predicting-steering-angle-in-a-self-driving-car)
  - [Table of Contents](#table-of-contents)
  - [1. Project Overview](#1-project-overview)
  - [2. Project Structure](#2-project-structure)
  - [3. Setup](#3-setup)
    - [3.1 Prerequisites](#31-prerequisites)
    - [3.2 Automated Setup](#32-automated-setup)
    - [3.3 Manual Setup](#33-manual-setup)
    - [3.4 Dataset](#34-dataset)
  - [4. Pipeline](#4-pipeline)
    - [4.1 Data Loading and Preprocessing](#41-data-loading-and-preprocessing)
    - [4.2 Sequence Construction](#42-sequence-construction)
    - [4.3 Model Architecture](#43-model-architecture)
    - [4.4 Training](#44-training)
    - [4.5 Inference and Visualization](#45-inference-and-visualization)
    - [4.6 Lane-Change Detection](#46-lane-change-detection)
  - [5. Module Reference](#5-module-reference)
    - [5.1 config.py](#51-configpy)
    - [5.2 data\_loader.py](#52-data_loaderpy)
    - [5.3 model.py](#53-modelpy)
    - [5.4 train.py](#54-trainpy)
    - [5.5 predict.py](#55-predictpy)
    - [5.6 lane\_change.py](#56-lane_changepy)
    - [5.7 utils.py](#57-utilspy)
  - [6. Usage](#6-usage)
    - [6.1 Training](#61-training)
    - [6.2 Inference](#62-inference)
  - [7. Configuration Reference](#7-configuration-reference)

---

## 1. Project Overview

This project implements an end-to-end steering angle prediction system. Instead of predicting from a single frame, the model takes a **sequence of consecutive frames** (default: 5) and uses an LSTM to capture temporal dynamics — how the road scene evolves over time — to produce a more stable and accurate steering prediction.

Key features:

- **CNN + LSTM architecture** — TimeDistributed convolutional layers extract spatial features per frame, then an LSTM fuses them over time.
- **In-memory image cache** — all images are pre-loaded into RAM once, eliminating disk I/O during training.
- **Three-camera augmentation** — left, right, and centre camera views are used with a steering offset to simulate recovery driving.
- **Lane-change detection** — a sliding-window heuristic detects lane changes from the predicted steering signal.
- **Visual output** — annotated frames with a steering wheel overlay, angle readout, and lane-change warnings. Optional MP4 video export.

---

## 2. Project Structure

```
project/
├── outputs/                          # Training curves and annotated frames (generated)
├── saved_model/                      # Trained model checkpoints (generated)
│   ├── best_model.keras
│   └── final_model.keras
├── self_driving_car_dataset_jungle/  # First dataset (images + CSV)
│   ├── IMG/
│   └── driving_log.csv
├── self_driving_car_dataset_make/    # Second dataset (images + CSV)
│   ├── IMG/
│   └── driving_log.csv
├── .gitignore
├── config.py                         # All hyperparameters, paths, and constants
├── data_loader.py                    # CSV parsing, image preprocessing, caching, Keras generator
├── lane_change.py                    # Lane-change detection from steering angle sequences
├── model.py                          # CNN + LSTM model definition (Keras Functional API)
├── predict.py                        # Inference script with overlays and video output
├── train.py                          # Training loop with callbacks and loss plotting
├── utils.py                          # Visualization helpers (steering wheel, warnings, video writer)
├── requirements.txt                  # Python dependencies
├── setup.ps1                         # Windows setup script (PowerShell)
├── setup.sh                          # Linux/macOS setup script (Bash)
└── README.md
```

---

## 3. Setup

### 3.1 Prerequisites

- **Python 3.10+**
- A dataset folder (e.g. `self_driving_car_dataset_jungle/`) containing a `driving_log.csv` and an `IMG/` directory with dashcam frames. The CSV must have columns: `center, left, right, steering, throttle, brake, speed`.

### 3.2 Automated Setup

**Windows (PowerShell):**

```powershell
cd project
.\setup.ps1
```

**Linux / macOS:**

```bash
cd project
./setup.sh
```

Both scripts create a virtual environment, activate it, and install all dependencies from `requirements.txt`.

### 3.3 Manual Setup

```bash
cd project
python -m venv venv

# Activate:
# Windows:  .\venv\Scripts\Activate.ps1
# Linux:    source venv/bin/activate

pip install -r requirements.txt
```

### 3.4 Dataset

Place the dataset folder inside the `project/` directory. The expected default name is `self_driving_car_dataset_jungle/`. If your dataset folder has a different name, update the `DATA_DIR` path in `config.py`.

The folder structure must be:

```
self_driving_car_dataset_jungle/
├── driving_log.csv
└── IMG/
    ├── center_2024_01_01_00_00_00_000.jpg
    ├── left_2024_01_01_00_00_00_000.jpg
    ├── right_2024_01_01_00_00_00_000.jpg
    └── ...
```

---

## 4. Pipeline

### 4.1 Data Loading and Preprocessing

The driving log CSV is parsed to extract image paths and steering angles. Each row in the CSV represents one timestep with three camera views (centre, left, right). The left and right images are assigned an adjusted steering angle (`± STEERING_OFFSET`, default 0.2) to teach the model recovery behaviour.

Each image is preprocessed as follows:

1. **Crop** — the top 60 pixels (sky) and bottom 25 pixels (car hood) are removed.
2. **Resize** — the cropped image is resized to 66×200 pixels using `cv2.INTER_AREA` (area-based interpolation, ideal for downscaling).
3. **Normalize** — pixel values are converted to `float32` and divided by 255.0 to produce values in [0, 1].

### 4.2 Sequence Construction

Rather than feeding single frames, the model receives **sequences of consecutive frames**. This is controlled by two parameters:

- `SEQ_LEN` (default: 5) — number of frames per sequence.
- `SEQ_STRIDE` (default: 3) — step between the start of consecutive sequences.

For example, with `SEQ_LEN=5` and `SEQ_STRIDE=3`, the sequences are: frames [0–4], [3–7], [6–10], etc. The label for each sequence is the steering angle of the **last frame** in that sequence.

All unique images are loaded into an `ImageCache` (a single NumPy array in RAM) at startup. During training, the Keras generator (`SteeringSequence`) assembles batches by indexing into this cache — there is zero disk I/O per batch.

### 4.3 Model Architecture

The model is built with the Keras Functional API:

```
Input: (batch, SEQ_LEN, 66, 200, 3)
  │
  ├─ TimeDistributed Conv2D(24, 5×5, stride 2, ReLU)
  ├─ TimeDistributed Conv2D(36, 5×5, stride 2, ReLU)
  ├─ TimeDistributed Conv2D(48, 3×3, ReLU)
  ├─ TimeDistributed Dropout(0.25)
  ├─ TimeDistributed GlobalAveragePooling2D  →  (batch, SEQ_LEN, 48)
  │
  ├─ LSTM(96)  →  (batch, 96)
  ├─ Dropout(0.3)
  │
  ├─ Dense(50, ReLU)
  └─ Dense(1, linear)  →  steering angle
```

The `TimeDistributed` wrapper applies the same CNN independently to each frame in the sequence. The LSTM then processes the resulting feature vectors in temporal order, learning how the scene evolves. The final Dense layer outputs a single scalar: the predicted steering angle.

### 4.4 Training

Training is managed by `train.py` with the following setup:

- **Optimizer**: Adam (learning rate 1e-3).
- **Loss**: Mean Squared Error (MSE) — standard for regression.
- **Metric**: Mean Absolute Error (MAE) — reported alongside loss for interpretability.
- **Callbacks**:
  - `EarlyStopping` — stops training if validation loss does not improve for 5 epochs, restores the best weights.
  - `ReduceLROnPlateau` — halves the learning rate if validation loss stalls for 3 epochs.
  - `ModelCheckpoint` — saves the best model to `saved_model/best_model.keras`.
- **Data augmentation**: random horizontal flip (image mirrored, steering negated) applied during training only.

After training, loss and MAE curves are saved to `outputs/training_curves.png`.

### 4.5 Inference and Visualization

`predict.py` loads a trained model and runs it frame-by-frame on the dataset:

1. A sliding buffer of the last `SEQ_LEN` preprocessed frames is maintained.
2. Once the buffer is full, the model predicts a steering angle for each new frame.
3. Each frame is annotated with:
   - A **steering wheel overlay** (rotated by the predicted angle, colour-coded green/yellow/red by magnitude).
   - The **predicted angle** in degrees.
   - The **ground truth** angle for comparison.
   - A **red border and warning text** if a lane change is detected.
4. Annotated frames are saved to `outputs/frames/`. Optionally, an MP4 video is produced.

### 4.6 Lane-Change Detection

The `LaneChangeDetector` analyses the stream of predicted steering angles using two heuristics:

1. **Zero-crossing + magnitude**: if the steering angle crosses zero (changes sign) and the peak absolute angle within the sliding window exceeds `LANE_CHANGE_ANGLE_THRESHOLD` (default 0.15), a lane change is flagged.
2. **Sustained derivative**: if the absolute frame-to-frame change in steering angle exceeds `LANE_CHANGE_DERIVATIVE_THRESHOLD` (default 0.04) for at least `LANE_CHANGE_SUSTAINED_FRAMES` (default 5) consecutive frames, a lane change is flagged.

The direction is determined by the sign of the most recent angle.

---

## 5. Module Reference

### 5.1 config.py

Central configuration file. All paths, image dimensions, sequence parameters, training hyperparameters, lane-change thresholds, and visualization constants are defined here. Every other module imports from `config.py`, so changes propagate everywhere.

### 5.2 data_loader.py

Handles all data I/O and preprocessing:

- `fix_path()` — resolves raw CSV paths to absolute paths in the `IMG/` directory.
- `preprocess_image()` — crops, resizes, and normalizes a single image.
- `load_image()` — reads an image from disk and applies preprocessing.
- `load_csv()` — parses the driving log CSV, expands three camera views per row.
- `ImageCache` — pre-loads all unique images into a single NumPy array in RAM for zero-I/O access during training.
- `_build_sequences()` — creates overlapping index sequences with configurable stride.
- `get_train_val_data()` — orchestrates the full data pipeline: CSV → split → expand → sequences → cache. Returns everything needed for training.
- `SteeringSequence` — a `keras.utils.Sequence` subclass that serves batches from the in-memory cache, with optional horizontal-flip augmentation.

### 5.3 model.py

Defines the CNN + LSTM architecture using the Keras Functional API:

- `build_model()` — constructs and returns the model. Three TimeDistributed Conv2D layers extract spatial features per frame, GlobalAveragePooling2D compresses them, an LSTM captures temporal dynamics, and a Dense head outputs the steering angle.

### 5.4 train.py

Training orchestration:

- `train()` — loads data, builds the model, configures callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint), runs `model.fit()`, saves the final model, and plots loss curves.
- `_plot_history()` — generates and saves training/validation loss and MAE plots.
- CLI interface with `--max-samples` flag for quick smoke tests.

### 5.5 predict.py

Inference and visualization:

- `run_inference()` — loads a trained model, iterates through frames maintaining a sequence buffer, predicts steering angles, draws overlays, detects lane changes, and saves annotated output.
- CLI flags: `--model`, `--max-frames`, `--max-samples`, `--video`, `--live`.

### 5.6 lane_change.py

Lane-change detection:

- `LaneChangeDetector` — stateful class with a sliding window. Call `update(angle)` per frame; it returns `(detected: bool, direction: str | None)`.
- `detect_lane_changes()` — batch function that runs the detector over an array of angles and returns a list of `(frame_index, direction)` events.

### 5.7 utils.py

Visualization utilities used by `predict.py`:

- `draw_steering_overlay()` — draws a rotating steering wheel and angle text on a frame.
- `draw_lane_change_warning()` — draws a red border and direction label.
- `make_video_writer()` — creates an OpenCV `VideoWriter` for MP4 output.

---

## 6. Usage

### 6.1 Training

```bash
# Full training (all data, 50 epochs max)
python train.py

# Quick smoke test (5000 CSV rows)
python train.py --max-samples 5000
```

Outputs:
- `saved_model/best_model.keras` — best checkpoint by validation loss
- `saved_model/final_model.keras` — model at the end of training
- `outputs/training_curves.png` — loss and MAE plots

### 6.2 Inference

```bash
# Run inference with default model
python predict.py

# Produce an MP4 video, limit to 500 frames
python predict.py --video --max-frames 500

# Use a specific model, show live preview
python predict.py --model saved_model/final_model.keras --live

# Quick test
python predict.py --max-samples 1000 --max-frames 200
```

Outputs:
- `outputs/frames/frame_00000.jpg`, `frame_00001.jpg`, ... — annotated frames
- `outputs/prediction.mp4` — video (if `--video` flag is used)

---

## 7. Configuration Reference

All parameters are set in `config.py`. Key ones to tune:

| Parameter | Default | Description |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `"-1"` | Set to `"0"` to enable GPU training |
| `MAX_SAMPLES` | `None` | Limit CSV rows (set to 5000 for smoke tests) |
| `SEQ_LEN` | `5` | Frames per input sequence |
| `SEQ_STRIDE` | `3` | Step between consecutive sequences |
| `BATCH_SIZE` | `32` | Training batch size |
| `EPOCHS` | `50` | Maximum training epochs |
| `LEARNING_RATE` | `1e-3` | Adam optimizer learning rate |
| `ES_PATIENCE` | `5` | EarlyStopping patience (epochs) |
| `LR_FACTOR` | `0.5` | Factor to reduce LR on plateau |
| `LR_PATIENCE` | `3` | Epochs before reducing LR |
| `STEERING_OFFSET` | `0.2` | Offset added/subtracted for left/right cameras |
| `CROP_TOP` / `CROP_BOTTOM` | `60` / `25` | Pixels cropped from top (sky) and bottom (hood) |
| `LANE_CHANGE_ANGLE_THRESHOLD` | `0.15` | Min |angle| for zero-crossing detection |
| `LANE_CHANGE_DERIVATIVE_THRESHOLD` | `0.04` | Min |d(angle)/dt| for sustained-derivative detection |
