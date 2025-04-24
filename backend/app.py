import os
import tempfile

import librosa
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment

from denoise import denoise_combined
from ml.predict import predict_emotion_cnn, predict_emotion_rnn, predict_emotion_rf, predict_emotion_hmm

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def convert_to_wav(src_path):
    # Определяем расширение
    ext = os.path.splitext(src_path)[1].lower()
    if ext == '.wav':
        return src_path  # WAV не трогаем
    try:
        # Конвертируем в WAV через pydub
        audio = AudioSegment.from_file(src_path)
        tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio.export(tmp_wav.name, format='wav')
        return tmp_wav.name
    except Exception as e:
        raise ValueError(f"Ошибка при конвертации аудио: {str(e)}")


def audio_to_melspectrogram(filepath, sr=22050, n_mels=128):
    try:
        y, sr = librosa.load(filepath, sr=sr)
        # Комбинированное подавление шума
        y = denoise_combined(y, sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db
    except Exception as e:
        raise ValueError(f"Ошибка при создании спектрограммы: {str(e)}")


@app.route('/newRecord', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'Аудио файл не найден'}), 400

    audio_file = request.files['audio']
    model = request.form.get('model', '').lower()

    if not audio_file:
        return jsonify({'error': 'Пустой файл'}), 400

    # Сохраняем файл
    filename = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(filename)

    # Конвертация в WAV если нужно
    try:
        wav_path = convert_to_wav(filename)
        mel_spec = audio_to_melspectrogram(wav_path)
        mel_shape = mel_spec.shape
        emotion = None
        probs = None
        if model == 'cnn':
            emotion, probs = predict_emotion_cnn(mel_spec)
        elif model == 'rnn':
            emotion, probs = predict_emotion_rnn(mel_spec)
        elif model == 'random forest':
            emotion, probs = predict_emotion_rf(mel_spec)
        elif model == 'hmm':
            emotion, probs = predict_emotion_hmm(mel_spec)
        if wav_path != filename:
            os.remove(wav_path)  # Удаляем временный файл
    except Exception as e:
        return jsonify({'error': f'Ошибка при обработке аудио: {str(e)}'}), 500

    return jsonify({
        'success': True,
        'model': model,
        'message': 'Файл успешно обработан',
        'mel_shape': mel_shape,
        'emotion': emotion,
        'probs': probs
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
