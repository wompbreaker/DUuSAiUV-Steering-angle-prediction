"""
CNN + LSTM model for steering angle prediction.

Architecture (Keras Functional API):
    - TimeDistributed CNN feature extractor (NVIDIA PilotNet-inspired, small)
    - LSTM temporal processor
    - Dense regression head  ->  single float (steering angle)
"""
import logging
# Import config first to set environment variables before TF/Keras
from config import SEQ_LEN, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS

import keras
from keras import layers


def build_model(
    seq_len: int = SEQ_LEN,
    img_h: int = IMG_HEIGHT,
    img_w: int = IMG_WIDTH,
    img_c: int = IMG_CHANNELS
) -> keras.Model:

    inputs = keras.Input(shape=(seq_len, img_h, img_w, img_c), name="input_seq")

    # --- TimeDistributed CNN feature extractor --- #
    # Conv block 1
    x = layers.TimeDistributed(
        layers.Conv2D(24, (5, 5), strides=(2, 2), activation="relu", padding="valid"),
        name="td_conv1"
    )(inputs)

    # Conv block 2
    x = layers.TimeDistributed(
        layers.Conv2D(36, (5, 5), strides=(2, 2), activation="relu", padding="valid"),
        name="td_conv2"
    )(x)

    # Conv block 3
    x = layers.TimeDistributed(
        layers.Conv2D(48, (3, 3), activation="relu", padding="valid"),
        name="td_conv3"
    )(x)

    x = layers.TimeDistributed(layers.Dropout(0.25), name="td_drop")(x)  # reduce overfitting

    # Global average pooling per frame - compact feature vector (48,)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D(), name="td_gap")(x)

    # --- LSTM temporal processor --- #
    x = layers.LSTM(96, return_sequences=False, name="lstm")(x)
    x = layers.Dropout(0.3, name="drop_lstm")(x)

    # --- Regression head --- #
    x = layers.Dense(50, activation="relu", name="fc1")(x)
    outputs = layers.Dense(1, activation="linear", name="steering_out")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="steering_cnn_lstm")
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
    total_params = model.count_params()
    logging.info(f"\nTotal parameters: {total_params:,}")
