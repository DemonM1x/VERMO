import os
import numpy as np
import librosa
from hmmlearn import hmm
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

spec_dir = '../data/specs'
emotion_mfcc = {}
emotions = set()

# Собираем MFCC для каждого класса
for file in tqdm(os.listdir(spec_dir), desc='Сбор MFCC'):
    if file.endswith('.npy'):
        label = file.split('_')[0]
        spec_path = os.path.join(spec_dir, file)
        mel_db = np.load(spec_path)
        # Приводим к двумерному виду
        if mel_db.ndim == 3:
            mel_db = mel_db[:, :, 0]
        elif mel_db.ndim == 1:
            mel_db = mel_db[np.newaxis, :]
        # Восстанавливаем сигнал из мел-спектрограммы (приближённо)
        S = librosa.db_to_power(mel_db)
        y_inv = librosa.feature.inverse.mel_to_audio(S, sr=22050, n_iter=8)
        mfcc = librosa.feature.mfcc(y=y_inv, sr=22050, n_mfcc=13).T  # (frames, 13)
        if label not in emotion_mfcc:
            emotion_mfcc[label] = []
        emotion_mfcc[label].append(mfcc)
        emotions.add(label)

# Обучаем HMM для каждой эмоции
hmm_models = {}
for label in tqdm(emotions, desc='Обучение HMM'):
    X = np.vstack(emotion_mfcc[label])  # (total_frames, 13)
    lengths = [mfcc.shape[0] for mfcc in emotion_mfcc[label]]
    model = hmm.GaussianHMM(n_components=6, covariance_type='diag', n_iter=100)
    model.fit(X, lengths)
    hmm_models[label] = model
    # Визуализация распределения MFCC
    plt.figure(figsize=(10, 4))
    plt.boxplot(X, vert=False, patch_artist=True)
    plt.title(f'MFCC распределение для {label}')
    plt.xlabel('MFCC коэффициент')
    plt.ylabel('Значение')
    plt.tight_layout()
    plt.savefig(f'mfcc_boxplot_{label}.png')
    plt.close()
    print(f'[{label}] Средний log-likelihood: {model.score(X):.2f}')

emotions = sorted(list(emotions))
joblib.dump({'models': hmm_models, 'emotions': emotions}, 'models/hmm_emotions.pkl')
