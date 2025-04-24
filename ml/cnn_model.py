import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

EMOTIONS = ['Angry', 'Happy', 'Sad', 'Neutral', 'Fearful', 'Disgusted', 'Surprised']


def build_cnn_model(input_shape=(128, 128, 1), num_classes=8):
    model = keras.models.Sequential([
        # Первый блок свертки
        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Второй блок свертки
        keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Третий блок свертки
        keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),
        
        # Полносвязный блок
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Компиляция с оптимизированными параметрами
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, 
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model


def train_cnn_model(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2):
    # Создаем модель
    model = build_cnn_model(input_shape=X_train.shape[1:], num_classes=len(EMOTIONS))
    
    # Добавляем callback для ранней остановки
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True,
        verbose=1
    )
    
    # Обучаем модель
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping]
    )
    
    return model, history


def load_cnn_model(model_path='models/cnn_emotions.keras', input_shape=(128, 128, 1)):
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception:
        # Если нет обученной модели — создаём новую (заглушка)
        model = build_cnn_model(input_shape=input_shape)
    return model
