import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

EMOTIONS = ['Angry', 'Happy', 'Sad', 'Neutral', 'Fearful', 'Disgusted', 'Surprised']


def build_rnn_model(input_shape=(128, 128, 1), num_classes=8):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Reshape((-1, 64)),  # (128,128,1) -> (30, 64) после сверток и пуллинга
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def load_rnn_model(model_path='models/rnn_emotions.keras', input_shape=(128, 128, 1)):
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception:
        model = build_rnn_model(input_shape=input_shape)
    return model
