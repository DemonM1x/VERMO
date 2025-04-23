import os
import librosa
import numpy as np
import csv

spec_dir = 'data/specs'
os.makedirs(spec_dir, exist_ok=True)
labels_path = os.path.join(spec_dir, 'labels.csv')

with open(labels_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'label'])

    for root, dirs, files in os.walk('data/emotions'):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3'):
                path = os.path.join(root, file)
                # Извлекаем метку из имени файла или структуры папок
                label = ...  # например, file.split('_')[0]
                mel = librosa.feature.melspectrogram(
                    y=librosa.load(path, sr=22050)[0], sr=22050, n_mels=128)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mel_db = mel_db[:, :128] if mel_db.shape[1] >= 128 else np.pad(mel_db, ((0, 0), (0, 128 - mel_db.shape[1])), mode='constant')
                spec_filename = os.path.splitext(file)[0] + '.npy'
                np.save(os.path.join(spec_dir, spec_filename), mel_db)
                writer.writerow([spec_filename, label])

