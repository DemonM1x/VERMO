import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from ml.rnn_model import build_rnn_model, train_rnn_model, EMOTIONS
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Константы
spec_dir = '../data/specs'
labels_path = os.path.join(spec_dir, 'labels.csv')
TARGET_SHAPE = (128, 128)
MODEL_PATH = 'models/rnn_emotions.keras'
EPOCHS = 30
BATCH_SIZE = 32
VAL_SPLIT = 0.2

# Загрузка данных из labels.csv
filename_to_label = {}
with open(labels_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        filename_to_label[row['filename']] = row['label']

X = []
y = []
emotions = set()

# Загрузка спектрограмм
print("Загрузка спектрограмм для RNN...")
i = 0
for file in os.listdir(spec_dir):
    if file.endswith('.npy') and file in filename_to_label:
        if i % 1000 == 0:
            print(f"Загружено {i} файлов")
        i += 1
        
        spec_path = os.path.join(spec_dir, file)
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
        
        # Нормализация
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        
        X.append(mel_db)
        label = filename_to_label[file]
        y.append(label)
        emotions.add(label)

# Преобразование данных
print("Подготовка данных...")
emotions = sorted(list(emotions))
X = np.array(X)[..., np.newaxis]  # Добавляем канал для CNN
y_indices = np.array([emotions.index(lbl) for lbl in y])

if len(X) < 2:
    raise ValueError('Недостаточно данных для обучения модели!')

# Перемешивание данных
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y_indices = y_indices[indices]

# Вычисление весов для балансировки классов
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_indices),
    y=y_indices
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Веса классов:", class_weights_dict)

# Создание генератора для аугментации данных
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=VAL_SPLIT
)

# Получение обучающей и валидационной выборки
train_generator = datagen.flow(
    X, y_indices,
    batch_size=BATCH_SIZE,
    subset='training'
)

validation_generator = datagen.flow(
    X, y_indices,
    batch_size=BATCH_SIZE,
    subset='validation'
)

# Callback для уменьшения скорости обучения
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

# Обучение модели с новой архитектурой
print(f"Начало обучения RNN на {len(X)} примерах...")
model = build_rnn_model(input_shape=(128, 128, 1), num_classes=len(emotions))

# Обучение с генератором данных
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[reduce_lr],
    class_weight=class_weights_dict
)

# Сохранение модели
model.save(MODEL_PATH)
print(f"Модель сохранена: {MODEL_PATH}")

# Визуализация процесса обучения
plt.figure(figsize=(15, 5))

# График точности
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Точность RNN модели')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()

# График функции потерь
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Функция потерь RNN модели')
plt.xlabel('Эпоха')
plt.ylabel('Потеря')
plt.legend()

plt.tight_layout()
plt.savefig('models/rnn_training_history.png')
print("График процесса обучения сохранен в models/rnn_training_history.png")

# Вывод итоговых метрик
print("\nИтоговые метрики RNN:")
print(f"Точность на обучающей выборке: {max(history.history['accuracy']):.4f}")
print(f"Точность на валидационной выборке: {max(history.history['val_accuracy']):.4f}")
print(f"Минимальная потеря на обучающей выборке: {min(history.history['loss']):.4f}")
print(f"Минимальная потеря на валидационной выборке: {min(history.history['val_loss']):.4f}")
