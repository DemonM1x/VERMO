import numpy as np
from ml.cnn_model import load_cnn_model, EMOTIONS
import tensorflow as tf


def predict_emotion_cnn(mel_spec, model=None):
    """
    mel_spec: np.ndarray (mel-спектрограмма, log-mel, shape (n_mels, time))
    model: keras.Model (если None — загружается автоматически)
    Возвращает: строка — название эмоции
    """
    # Приводим к размеру (128, 128) и нормализуем
    mel = mel_spec
    mel = mel[:128, :128] if mel.shape[0] >= 128 and mel.shape[1] >= 128 else np.pad(mel, (
        (0, max(0, 128 - mel.shape[0])), (0, max(0, 128 - mel.shape[1]))), mode='constant')
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)
    mel = mel[..., np.newaxis]  # (128, 128, 1)
    mel = np.expand_dims(mel, axis=0)  # (1, 128, 128, 1)

    if model is None:
        model = load_cnn_model()
    preds = model.predict(mel)
    idx = np.argmax(preds)
    return EMOTIONS[idx]
