import numpy as np
import librosa
import pywt

def spectral_subtract_denoise(y, sr, n_fft=2048, hop_length=512, noise_frames=10):
    """
    Подавление шума методом спектрального вычитания.
    Оценка шума по первым noise_frames окнам.
    Возвращает очищенный сигнал.
    """
    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(D), np.angle(D)

    # Оценка спектра шума по первым noise_frames
    noise_mag = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    # Спектральное вычитание
    magnitude_denoised = magnitude - noise_mag
    magnitude_denoised = np.maximum(magnitude_denoised, 0.0)

    # Восстановление сигнала
    D_denoised = magnitude_denoised * np.exp(1j * phase)
    y_denoised = librosa.istft(D_denoised, hop_length=hop_length, length=len(y))
    return y_denoised


def wavelet_denoise(y, wavelet='db8', level=4, threshold_factor=0.5):
    """
    Вейвлет-фильтрация аудиосигнала.
    threshold_factor — множитель для медианного порога детальных коэффициентов.
    """
    coeffs = pywt.wavedec(y, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = threshold_factor * sigma * np.sqrt(2 * np.log(len(y)))
    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        new_coeffs.append(pywt.threshold(c, value=uthresh, mode='soft'))
    y_denoised = pywt.waverec(new_coeffs, wavelet)
    return y_denoised[:len(y)]


def denoise_combined(y, sr):
    """
    Сначала спектральное вычитание, затем вейвлет-фильтрация.
    """
    y1 = spectral_subtract_denoise(y, sr)
    y2 = wavelet_denoise(y1)
    return y2 