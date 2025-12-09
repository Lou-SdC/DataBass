"""
Module to generate the mel spectrogramms from a librosa loaded file (y, sr) and to plot it
"""

import librosa
import matplotlib.pyplot as plt
import numpy as np

def generate_mel_spectrogram(y, sr, n_mels=128, fmax=8000, n_fft=2048, hop_length=512):
    """
    generate a mel-spectrogram from y, sr extracted from the audio file.

    Args:
        y (np.ndarray): the audio file extracted with librosa
        sr (int): the sampling rate of the audio file
        n_mels (int): number of Mel. bands. Default: 128.
        fmax (int): max frequency to use. Default: 8000 Hz.
        n_fft (int): size of the FFT window (higher = better frequencial resolution)
        hop_length (int): step between windows (smaller = better temporal resolution)

    Returns:
        np.ndarray: Mel-spectrogram (shape: [n_mels, time_steps]).
    """

    # create the Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        fmax=fmax,
        n_fft=2048,
        hop_length=512
    )

    # dB conversion
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram_db


def plot_mel_spectrogram(mel_spec, sr):
    """function to plot a spectrogram generated with generate_mel_spectrogram

    Args:
        mel_spec (np.ndarray): a Mel-spectrogram
        sr (int): the sampling rate of the audio file
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-spectrogram')
    plt.show()
