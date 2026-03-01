"""
Training script for the steering angle prediction model.

Usage:
    python train.py                 # train with defaults from config.py
    python train.py --max_samples 5000   # quick smoke-test on 5 000 rows
"""

import argparse
import logging
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib
matplotlib.use("Agg")
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# Ensure config env vars are set before any TF import
import config
from config import (
    SAVED_MODEL_DIR, OUTPUT_DIR,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    ES_PATIENCE, LR_FACTOR, LR_PATIENCE,
)
from data_loader import get_train_val_data, SteeringSequence
from model import build_model


def train(max_samples=None):
    # Optionally override MAX_SAMPLES for a quick run
    if max_samples is not None:
        config.MAX_SAMPLES = max_samples
        logging.info(f"MAX_SAMPLES overridden to {max_samples}")

    # --- 1. Prepare data --- #
    logging.info("Loading data...")
    cache, tr_idx, tr_labels, va_idx, va_labels = get_train_val_data()
    logging.info(f"Train sequences: {len(tr_labels)}")
    logging.info(f"Val sequences: {len(va_labels)}")

    train_gen = SteeringSequence(
        cache, tr_idx, tr_labels, batch_size=BATCH_SIZE, shuffle=True, augment=True
    )
    val_gen = SteeringSequence(
        cache, va_idx, va_labels, batch_size=BATCH_SIZE, shuffle=False, augment=False
    )

    # --- 2. Build & compile model --- #
    model = build_model()
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    model.summary()

    # --- 3. Callbacks --- #
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=ES_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=LR_FACTOR,
            patience=LR_PATIENCE, verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(SAVED_MODEL_DIR, "best_model.keras"),
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
    ]

    # --- 4. Train --- #
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # --- 5. Save final model --- #
    final_path = os.path.join(SAVED_MODEL_DIR, "final_model.keras")
    model.save(final_path)
    logging.info(f"Final model saved to {final_path}")

    # --- 6. Plot loss curves --- #
    _plot_history(history)

    return model, history


def _plot_history(history):
    """Save training / validation loss and MAE curves."""
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss (MSE)
    axes[0].plot(history.history["loss"], label="train loss")
    axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_title("MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # MAE
    axes[1].plot(history.history["mae"], label="train MAE")
    axes[1].plot(history.history["val_mae"], label="val MAE")
    axes[1].set_title("Mean Absolute Error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"Training curves saved to {out_path}")


# --- CLI entry point --- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train steering angle model")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit the number of CSV rows used (for quick testing).",
    )
    args = parser.parse_args()
    train(max_samples=args.max_samples)
