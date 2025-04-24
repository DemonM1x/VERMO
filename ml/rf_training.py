import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import librosa
import csv

spec_dir = '../data/specs'
labels_path = os.path.join(spec_dir, 'labels.csv')

# Читаем метки из labels.csv
filename_to_label = {}
with open(labels_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        filename_to_label[row['filename']] = row['label']

X = []
y = []
emotions = set()

for file in os.listdir(spec_dir):
    if file.endswith('.npy') and file in filename_to_label:
        label = filename_to_label[file]
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
y_indices = np.array([emotions.index(lbl) for lbl in y])

if len(X) < 2:
    raise ValueError('Недостаточно данных для обучения модели!')
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=16)
clf.fit(X, y_indices)
joblib.dump({'model': clf, 'emotions': emotions}, 'models/rf_emotions.pkl')
