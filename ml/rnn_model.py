import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

EMOTIONS = ['Angry', 'Happy', 'Sad', 'Neutral', 'Fearful', 'Disgusted', 'Surprised']


def build_rnn_model(input_shape=(128, 128, 1), num_classes=8):
    model = models.Sequential([
        # Первый сверточный блок
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Второй сверточный блок
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Преобразование для RNN слоя
        layers.Reshape((-1, 64)),  # После сверток и пуллинга
        
        # RNN блок
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(0.3),
        
        # Полносвязный блок
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Компиляция с оптимизированными параметрами
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, 
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model


def train_rnn_model(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2):
    # Создаем модель
    model = build_rnn_model(input_shape=X_train.shape[1:], num_classes=len(EMOTIONS))
    
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


def load_rnn_model(model_path='models/rnn_emotions.keras', input_shape=(128, 128, 1)):
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception:
        model = build_rnn_model(input_shape=input_shape)
    return model
