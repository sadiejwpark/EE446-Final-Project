"""
Two small architectures for comparison (rubric: "multiple models/approaches"
advanced component):

1. DS-CNN  -- depthwise-separable CNN over the [frames, mfcc] "image",
              in the style of Zhang et al. "Hello Edge" small DS-CNN.
              This is the primary/deployed model.
2. CNN-1D  -- 1D conv over time, treating each frame's MFCC vector as a
              channel. Smaller and faster, used as the comparison baseline.

Both take input shape [NUM_FRAMES, NUM_MFCC] and output NUM_CLASSES softmax.
Keep parameter counts small (tens of thousands) so INT8 models comfortably
fit Nano 33 BLE Sense's ~1MB flash / 256KB RAM budget.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from . import config as cfg


def build_ds_cnn(num_frames=None, num_mfcc=None, num_classes=None, width_multiplier=1.0):
    num_frames = num_frames or cfg.NUM_FRAMES
    num_mfcc = num_mfcc or cfg.NUM_MFCC
    num_classes = num_classes or cfg.NUM_CLASSES

    def c(filters):
        return max(4, int(filters * width_multiplier))

    inputs = layers.Input(shape=(num_frames, num_mfcc, 1), name="mfcc_input")

    x = layers.Conv2D(c(64), kernel_size=(10, 4), strides=(2, 2), padding="same",
                       use_bias=False, name="conv_stem")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for i, filters in enumerate([c(64), c(64), c(64), c(64)]):
        x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same",
                                    use_bias=False, name=f"dw_{i}")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, kernel_size=1, strides=1, padding="same",
                           use_bias=False, name=f"pw_{i}")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return models.Model(inputs, outputs, name="ds_cnn")


def build_cnn_1d(num_frames=None, num_mfcc=None, num_classes=None):
    num_frames = num_frames or cfg.NUM_FRAMES
    num_mfcc = num_mfcc or cfg.NUM_MFCC
    num_classes = num_classes or cfg.NUM_CLASSES

    inputs = layers.Input(shape=(num_frames, num_mfcc), name="mfcc_input")  # [time, channels]

    x = layers.Conv1D(48, kernel_size=5, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(48, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(64, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return models.Model(inputs, outputs, name="cnn_1d")


def count_params(model):
    return model.count_params()


if __name__ == "__main__":
    ds = build_ds_cnn()
    ds.summary()
    print("DS-CNN params:", count_params(ds))

    c1d = build_cnn_1d()
    c1d.summary()
    print("CNN-1D params:", count_params(c1d))
