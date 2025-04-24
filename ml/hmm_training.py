import os
import numpy as np
import librosa
from hmmlearn import hmm
import joblib
from tqdm import tqdm
import csv

spec_dir = '../data/specs'
labels_path = os.path.join(spec_dir, 'labels.csv')

# Читаем метки из labels.csv
filename_to_label = {}
with open(labels_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        filename_to_label[row['filename']] = row['label']

emotion_mfcc = {}
emotions = set()

# Собираем MFCC для каждого класса
for file in tqdm(os.listdir(spec_dir), desc='Сбор MFCC'):
    if file.endswith('.npy') and file in filename_to_label:
        label = filename_to_label[file]
        spec_path = os.path.join(spec_dir, file)
        mel_db = np.load(spec_path)
        if mel_db.ndim == 3:
            mel_db = mel_db[:, :, 0]
        elif mel_db.ndim == 1:
            mel_db = mel_db[np.newaxis, :]
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
    if len(emotion_mfcc[label]) < 2:
        print(f'Пропуск {label}: слишком мало файлов')
        continue
    X = np.vstack(emotion_mfcc[label])  # (total_frames, 13)
    lengths = [mfcc.shape[0] for mfcc in emotion_mfcc[label]]
    if X.shape[0] < 20 or min(lengths) < 5:
        print(f'Пропуск {label}: слишком мало кадров')
        continue
    try:
        n_components = min(6, max(2, X.shape[0] // 20))
        model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=100)
        model.fit(X, lengths)
        hmm_models[label] = model
        print(f'[{label}] Средний log-likelihood: {model.score(X):.2f}')
    except Exception as e:
        print(f'Ошибка при обучении HMM для {label}: {e}')

emotions = sorted(list(hmm_models.keys()))
joblib.dump({'models': hmm_models, 'emotions': emotions}, 'models/hmm_emotions.pkl')
