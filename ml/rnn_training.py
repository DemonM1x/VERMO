import csv
import os
import numpy as np
from ml.rnn_model import build_rnn_model, EMOTIONS

spec_dir = '../data/specs'
labels_path = os.path.join(spec_dir, 'labels.csv')

X = []
y = []
emotions = set()

with open(labels_path, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        spec_path = os.path.join(spec_dir, row['filename'])
        if os.path.exists(spec_path):
            X.append(np.load(spec_path))
            y.append(row['label'])
            emotions.add(row['label'])

emotions = sorted(list(emotions))
X = np.array(X)[..., np.newaxis]
y = np.array([emotions.index(lbl) for lbl in y])

if len(X) < 2:
    raise ValueError('Недостаточно данных для обучения модели!')

model = build_rnn_model(input_shape=(128, 128, 1), num_classes=len(EMOTIONS))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
model.save('models/rnn_emotions.keras')
