import logging
import os
from typing import Tuple
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import keras
import numpy as np
import pandas as pd

from config import (
    CSV_PATH, IMG_DIR,
    CROP_TOP, CROP_BOTTOM,
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
    STEERING_OFFSET, SEQ_LEN, SEQ_STRIDE,
    VAL_SPLIT, MAX_SAMPLES, SEED, BATCH_SIZE,
)


def fix_path(raw: str) -> str:
    """Convert a raw path from the CSV to an absolute path in IMG_DIR.

    Parameters
    ----------
    raw : str
        Raw path from CSV

    Returns
    -------
    str
        Absolute path to the image file in IMG_DIR
    """
    fname = raw.strip()
    # Take only the filename part after the last separator
    basename = os.path.basename(fname)
    return os.path.join(IMG_DIR, basename)


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Crop, resize, and normalise an image for model input.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR format (as read by cv2.imread).

    Returns
    -------
    np.ndarray
        Preprocessed image in RGB format, resized and normalized to [0, 1].
    """
    # Crop sky and car hood
    cropped = img[CROP_TOP: img.shape[0] - CROP_BOTTOM, :, :]
    # Resize to target dimensions
    resized = cv2.resize(cropped, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    # Normalise to [0, 1]
    return resized.astype(np.float32) / 255.0


def load_image(path: str) -> np.ndarray:
    """Load an image from disk (BGR), apply preprocessing

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        Preprocessed image ready for model input.

    Raises
    ------
    FileNotFoundError
        If the image file cannot be found or read.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return preprocess_image(img)


def load_csv() -> pd.DataFrame:
    """Load the driving log CSV and expand it to a DataFrame of (image_path, steering) pairs.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["image_path", "steering"] where each row corresponds to
        a single image (center, left, or right) and its associated steering angle.
    """
    columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
    dataframe = pd.read_csv(CSV_PATH, names=columns, skiprows=1)

    if MAX_SAMPLES is not None:
        dataframe = dataframe.iloc[:MAX_SAMPLES]

    rows = []
    for _, row in dataframe.iterrows():
        rows.append((fix_path(row["center"]), float(row["steering"])))
        rows.append((fix_path(row["left"]), float(row["steering"]) + STEERING_OFFSET))
        rows.append((fix_path(row["right"]), float(row["steering"]) - STEERING_OFFSET))

    out = pd.DataFrame(rows, columns=["image_path", "steering"])
    out["steering"] = out["steering"].clip(-1.0, 1.0)
    return out


class ImageCache:
    """
    Loads all unique images into a single numpy array at construction time.
    Subsequent access is by integer index — zero disk I/O during training.
    """

    def __init__(self, paths: np.ndarray) -> None:
        """
        Parameters
        ----------
        paths : 1-D array of file-path strings (may contain duplicates).
        """
        # De-duplicate to minimise reads
        unique_paths, inverse = np.unique(paths, return_inverse=True)
        n = len(unique_paths)
        self._data = np.empty(
            (n, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32
        )
        logging.info(f"Pre-loading {n} unique images "
              f"({n * IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS * 4 / 1e6:.0f} MB) ...")
        for i, p in enumerate(unique_paths):
            self._data[i] = load_image(p)
            if (i + 1) % 2000 == 0:
                logging.info(f"    {i + 1}/{n} loaded")
        # Map the original ordering to unique-array indices
        self._idx = inverse
        logging.info(f"Image cache ready.")

    def __getitem__(self, original_index: int) -> np.ndarray:
        return self._data[self._idx[original_index]]

    def get_batch(self, indices: np.ndarray) -> np.ndarray:
        """Get a batch of images by their original indices.

        Parameters
        ----------
        indices : np.ndarray
            Array of integer indices corresponding to the original ordering of paths.

        Returns
        -------
        np.ndarray
            Batch of preprocessed images corresponding to the input indices.
        """
        return self._data[self._idx[indices]]


def _build_sequences(
    n_images: int,
    steerings: np.ndarray,
    seq_len: int = SEQ_LEN,
    stride: int = SEQ_STRIDE
) -> Tuple[np.ndarray, np.ndarray]:
    """Build overlapping sequences of integer indices with a configurable stride.

    The label for each sequence is the steering angle of the last frame in that sequence.

    Parameters
    ----------
    n_images : int
        Total number of images (length of the dataset).
    steerings : np.ndarray
        Array of steering angles corresponding to each image.
    seq_len : int, optional
        Length of each sequence, by default SEQ_LEN
    stride : int, optional
        Step size between the start of consecutive sequences, by default SEQ_STRIDE

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - seq_indices: Array of shape (N, seq_len) containing integer indices for each sequence.
        - seq_labels: Array of shape (N,) containing the steering angle label for each sequence
    """
    starts = np.arange(0, n_images - seq_len + 1, stride)
    seq_indices = np.array([np.arange(s, s + seq_len) for s in starts])
    seq_labels = steerings[starts + seq_len - 1]
    return seq_indices, seq_labels


def get_train_val_data() -> Tuple[ImageCache, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the dataset, returning an ImageCache and train/val sequence indices and labels.

    Given the CSV log, this function:
    - Loads and expands the CSV into a DataFrame of (image_path, steering) pairs
    - Builds overlapping sequences of integer indices for training and validation sets
    - Pre-loads all unique images into an ImageCache for fast access during training

    Returns
    -------
    Tuple[ImageCache, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - ImageCache: An instance of ImageCache containing all preprocessed images
        - train_seq_idx: Array of shape (N_train, SEQ_LEN) with integer indices for training sequences
        - train_seq_labels: Array of shape (N_train,) with steering angle labels for training sequences
        - val_seq_idx: Array of shape (N_val, SEQ_LEN) with integer indices for validation sequences
        - val_seq_labels: Array of shape (N_val,) with steering angle labels for validation sequences
    """
    columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
    raw_dataframe = pd.read_csv(CSV_PATH, names=columns, skiprows=1)
    if MAX_SAMPLES is not None:
        raw_dataframe = raw_dataframe.iloc[:MAX_SAMPLES]

    n = len(raw_dataframe)
    split_point = int(n * (1 - VAL_SPLIT))

    def _expand(sub_df: pd.DataFrame):
        paths, steers = [], []
        for _, row in sub_df.iterrows():
            paths.append(fix_path(row["center"]))
            steers.append(float(row["steering"]))
            paths.append(fix_path(row["left"]))
            steers.append(float(row["steering"]) + STEERING_OFFSET)
            paths.append(fix_path(row["right"]))
            steers.append(float(row["steering"]) - STEERING_OFFSET)
        steers = np.clip(steers, -1.0, 1.0)
        return np.array(paths), np.array(steers, dtype=np.float32)

    train_paths, train_steers = _expand(raw_dataframe.iloc[:split_point])
    val_paths, val_steers = _expand(raw_dataframe.iloc[split_point:])

    # Build sequences
    train_seq_idx, train_seq_labels = _build_sequences(len(train_paths), train_steers)
    val_seq_idx, val_seq_labels = _build_sequences(len(val_paths), val_steers)

    # Pre-load ALL images once into RAM
    all_paths = np.concatenate([train_paths, val_paths])
    logging.info(
        f"Total images: {len(all_paths)} "
        f"(train {len(train_paths)}, val {len(val_paths)})"
    )
    cache = ImageCache(all_paths)

    # Shift val indices so they reference the global cache correctly
    val_seq_idx = val_seq_idx + len(train_paths)

    return cache, train_seq_idx, train_seq_labels, val_seq_idx, val_seq_labels


class SteeringSequence(keras.utils.Sequence):
    """Fast generator backed by an in-memory ImageCache."""

    def __init__(
        self,
        cache: ImageCache,
        seq_idx: np.ndarray,
        labels: np.ndarray,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
        augment: bool = False
    ) -> None:
        super().__init__()
        self.cache: ImageCache = cache
        self.seq_idx: np.ndarray = seq_idx
        self.labels: np.ndarray = labels
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.augment: bool = augment
        self.indices: np.ndarray = np.arange(len(self.labels))
        if self.shuffle:
            np.random.seed(SEED)
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))

    def on_epoch_end(self) -> None:
        """Shuffle the sequence indices at the end of each epoch if shuffle is enabled."""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_idx = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_size = len(batch_idx)
        batch_x = np.empty(
            (batch_size, SEQ_LEN, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32
        )
        batch_y = self.labels[batch_idx].copy()

        for batch_position, sequence_index in enumerate(batch_idx):
            for time_step in range(SEQ_LEN):
                batch_x[batch_position, time_step] = self.cache[self.seq_idx[sequence_index, time_step]]

            if self.augment and np.random.random() > 0.5:
                batch_x[batch_position] = batch_x[batch_position, :, :, ::-1, :]   # flip W axis
                batch_y[batch_position] = -batch_y[batch_position]

        return batch_x, batch_y


if __name__ == "__main__":
    logging.info("Loading data ...")
    cache, tr_i, tr_l, va_i, va_l = get_train_val_data()
    logging.info(f"Train sequences: {tr_i.shape[0]}, Val sequences: {va_i.shape[0]}")
    logging.info(f"Sequence length: {SEQ_LEN}, stride: {SEQ_STRIDE}")
    logging.info(f"Label range: [{tr_l.min():.3f}, {tr_l.max():.3f}]")

    gen = SteeringSequence(cache, tr_i, tr_l, batch_size=4, shuffle=False)
    x, y = gen[0]
    logging.info(f"Batch X shape: {x.shape}")
    logging.info(f"Batch Y shape: {y.shape}")
    logging.info("Data loader functionality test passed.")
