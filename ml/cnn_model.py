import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

EMOTIONS = ['angry', 'happy', 'sad', 'neutral', 'fear', 'disgust', 'surprise', 'calm']


def build_cnn_model(input_shape=(128, 128, 1), num_classes=8):
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def load_cnn_model(model_path='models/cnn_emotions.keras', input_shape=(128, 128, 1)):
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception:
        # Если нет обученной модели — создаём новую (заглушка)
        model = build_cnn_model(input_shape=input_shape)
    return model
