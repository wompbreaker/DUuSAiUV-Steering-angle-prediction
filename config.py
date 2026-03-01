"""
Configuration file for Steering Angle Prediction & Lane Change Detection.
All hyperparameters, paths, and constants are defined here.
"""

import os

# ------------------- #
# --- Environment --- #
# ------------------- #
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Force CPU training initially
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ------------- #
# --- Paths --- #
# ------------- #
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "self_driving_car_dataset_jungle")
CSV_PATH = os.path.join(DATA_DIR, "driving_log.csv")
IMG_DIR = os.path.join(DATA_DIR, "IMG")

SAVED_MODEL_DIR = os.path.join(PROJECT_DIR, "saved_model")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

# --------------------------- #
# --- Image preprocessing --- #
# --------------------------- #
CROP_TOP = 60
CROP_BOTTOM = 25
IMG_HEIGHT = 66
IMG_WIDTH = 200
IMG_CHANNELS = 3

# ---------------------------------- #
# --- Data / sequence parameters --- #
# ---------------------------------- #
# Steering offset applied to left / right camera images
STEERING_OFFSET = 0.2

# Number of consecutive frames forming one input sequence for the LSTM
SEQ_LEN = 5

# Stride for building sequences (1 = max overlap, SEQ_LEN = no overlap).
# Higher stride = fewer sequences = faster epoch, less redundancy.
SEQ_STRIDE = 3

# Fraction of data used for validation
VAL_SPLIT = 0.2

# Maximum number of CSV rows to use
MAX_SAMPLES = None

# Random seed
SEED = 42

# -------------------------------- #
# --- Training hyperparameters --- #
# -------------------------------- #
BATCH_SIZE = 32  # For CPU 
EPOCHS = 50
LEARNING_RATE = 1e-3

# EarlyStopping
ES_PATIENCE = 7
# ReduceLROnPlateau
LR_FACTOR = 0.5
LR_PATIENCE = 3

# ----------------------------- #
# --- Lane-change detection --- #
# ----------------------------- #
# Sliding window length (number of frames) for lane-change analysis
LANE_CHANGE_WINDOW = 20

# A lane change is flagged when the steering angle crosses zero AND
# the absolute angle exceeds this threshold at any point in the window.
LANE_CHANGE_ANGLE_THRESHOLD = 0.15

# OR when the moving-average |derivative| exceeds this value for
# LANE_CHANGE_SUSTAINED_FRAMES consecutive frames.
LANE_CHANGE_DERIVATIVE_THRESHOLD = 0.04
LANE_CHANGE_SUSTAINED_FRAMES = 5

# --------------------- #
# --- Visualization --- #
# --------------------- #
# Angle thresholds for colour coding the arrow overlay
VIS_STRAIGHT_THRESH = 0.1   # |angle| < this  -> green
VIS_MODERATE_THRESH = 0.4   # |angle| < this  -> yellow, else red
