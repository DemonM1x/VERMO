import numpy as np
from ml.cnn_model import load_cnn_model, EMOTIONS
from ml.rnn_model import load_rnn_model
import tensorflow as tf
import joblib
import os
import librosa

_rf_model = None
_rf_emotions = None
_hmm_models = None
_hmm_emotions = None

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

def predict_emotion_rnn(mel_spec, model=None):
    mel = mel_spec
    mel = mel[:128, :128] if mel.shape[0] >= 128 and mel.shape[1] >= 128 else np.pad(mel, ((0, max(0, 128-mel.shape[0])), (0, max(0, 128-mel.shape[1]))), mode='constant')
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)
    mel = mel[..., np.newaxis]
    mel = np.expand_dims(mel, axis=0)
    if model is None:
        model = load_rnn_model()
    preds = model.predict(mel)
    idx = np.argmax(preds)
    return EMOTIONS[idx]

def predict_emotion_rf(mel_spec):
    global _rf_model, _rf_emotions
    if _rf_model is None or _rf_emotions is None:
        model_path = os.path.join('models', 'rf_emotions.pkl')
        data = joblib.load(model_path)
        _rf_model = data['model']
        _rf_emotions = data['emotions']
    mel = mel_spec
    mel = mel[:128, :128] if mel.shape[0] >= 128 and mel.shape[1] >= 128 else np.pad(mel, ((0, max(0, 128-mel.shape[0])), (0, max(0, 128-mel.shape[1]))), mode='constant')
    feature_vec = mel.mean(axis=1).reshape(1, -1)
    idx = _rf_model.predict(feature_vec)[0]
    return _rf_emotions[idx]

def predict_emotion_hmm(mel_spec):
    global _hmm_models, _hmm_emotions
    if _hmm_models is None or _hmm_emotions is None:
        model_path = os.path.join('models', 'hmm_emotions.pkl')
        data = joblib.load(model_path)
        _hmm_models = data['models']
        _hmm_emotions = data['emotions']
    mel = mel_spec
    mel = mel[:128, :128] if mel.shape[0] >= 128 and mel.shape[1] >= 128 else np.pad(mel, ((0, max(0, 128-mel.shape[0])), (0, max(0, 128-mel.shape[1]))), mode='constant')
    S = librosa.db_to_power(mel)
    y_inv = librosa.feature.inverse.mel_to_audio(S, sr=22050, n_iter=8)
    mfcc = librosa.feature.mfcc(y=y_inv, sr=22050, n_mfcc=13).T  # (frames, 13)
    scores = [model.score(mfcc) for model in _hmm_models.values()]
    idx = int(np.argmax(scores))
    return _hmm_emotions[idx]
