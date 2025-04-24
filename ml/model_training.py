import os
import numpy as np
from ml.cnn_model import build_cnn_model
import csv

spec_dir = '../data/specs'
labels_path = os.path.join(spec_dir, 'labels.csv')
X = []
y = []
emotions = set()
TARGET_SHAPE = (128, 128)

i = 0
with open(labels_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        spec_path = os.path.join(spec_dir, row['filename'])
        if os.path.exists(spec_path):
            if i % 10000 == 0:
                print("iteration = " + str(i), end='\n')
            i += 1
            mel_db = np.load(spec_path)
            # Приводим к двумерному виду
            if mel_db.ndim == 3:
                mel_db = mel_db[:, :, 0]
            elif mel_db.ndim == 1:
                mel_db = mel_db[np.newaxis, :]
            # Приводим к TARGET_SHAPE
            h, w = mel_db.shape
            pad_h = max(0, TARGET_SHAPE[0] - h)
            pad_w = max(0, TARGET_SHAPE[1] - w)
            mel_db = mel_db[:TARGET_SHAPE[0], :TARGET_SHAPE[1]]
            if pad_h > 0 or pad_w > 0:
                mel_db = np.pad(mel_db, ((0, pad_h), (0, pad_w)), mode='constant')
            X.append(mel_db)
            y.append(row['label'])
            emotions.add(row['label'])

emotions = sorted(list(emotions))
X = np.array(X)[..., np.newaxis]
y = np.array([emotions.index(lbl) for lbl in y])

if len(X) < 2:
    raise ValueError('Недостаточно данных для обучения модели!')

model = build_cnn_model(input_shape=(128, 128, 1), num_classes=len(emotions))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
model.save('models/cnn_emotions.keras')
