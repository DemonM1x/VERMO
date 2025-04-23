import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import librosa

spec_dir = '../data/specs'
X = []
y = []
emotions = set()

for file in os.listdir(spec_dir):
    if file.endswith('.npy'):
        label = file.split('_')[0]
        spec_path = os.path.join(spec_dir, file)
        mel_db = np.load(spec_path)
        # Приводим к двумерному виду
        if mel_db.ndim == 3:
            mel_db = mel_db[:, :, 0]
        elif mel_db.ndim == 1:
            mel_db = mel_db[np.newaxis, :]
        # Усредняем по времени (ось 1), получаем вектор длины 128
        feature_vec = mel_db.mean(axis=1)
        # MFCC (используем librosa, преобразуем обратно в сигнал)
        # Восстанавливаем сигнал из мел-спектрограммы (приближённо)
        S = librosa.db_to_power(mel_db)
        y_inv = librosa.feature.inverse.mel_to_audio(S, sr=22050, n_iter=8)
        mfcc = librosa.feature.mfcc(y=y_inv, sr=22050, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)  # 13 признаков
        full_vec = np.concatenate([feature_vec, mfcc_mean])
        X.append(full_vec)
        y.append(label)
        emotions.add(label)

emotions = sorted(list(emotions))
X = np.array(X)
y = np.array([emotions.index(lbl) for lbl in y])

if len(X) < 2:
    raise ValueError('Недостаточно данных для обучения модели!')

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
joblib.dump({'model': clf, 'emotions': emotions}, 'models/rf_emotions.pkl')
