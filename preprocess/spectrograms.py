"""
Module to generate the mel spectrogramms from a librosa loaded file (y, sr) and to plot it
"""

import librosa
import matplotlib.pyplot as plt
import numpy as np

def generate_mel_spectrogram(y, sr, fmax=8000, n_fft=2048,
                             normalize=False, target_shape=(128, 128), duration=1.0):
    """
    generate a mel-spectrogram from y, sr extracted from the audio file.

    Args:
        y (np nd.array): the audio file extracted with librosa
        sr (int): the sampling rate of the audio file
        n_mels (int): number of Mel. bands. Default: 128.
        fmax (int): max frequency to use. Default: 8000 Hz.
        n_fft (int): size of the FFT window (higher = better frequencial resolution)
        normalize (bool or str): type of normalization you want to perform:
            False = no normalization, 'minmax' = MinMax normalization,
            'standard' = standard normalization
        target_shape (tuple): target size (height * width). default (128, 128)
        duration (float): duration of the audio file to analyze. default 2 sec

    Returns:
        np.ndarray: Mel-spectrogram (shape: [n_mels, time_steps]).
    """
    # Compute hop_length to get target size
    n_mels, width_target = target_shape
    # step between windows
    hop_length = int((duration * sr) / (width_target - 1))


    # create the Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        fmax=fmax,
        n_fft=2048,
        hop_length=hop_length
    )

    # dB conversion
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    if normalize==False:
        return mel_spectrogram_db

    elif normalize=='minmax':
        mel_spectrogram_db_minmax = (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min())
        return mel_spectrogram_db_minmax

    elif normalize=='standard':
        mel_spectrogram_db_standard = (mel_spectrogram_db - mel_spectrogram_db.mean()) / mel_spectrogram_db.std()
        return mel_spectrogram_db_standard


def plot_mel_spectrogram(mel_spec, sr, target_shape=(128,128)):
    """function to plot a spectrogram generated with generate_mel_spectrogram

    Args:
        mel_spec (np.ndarray): a Mel-spectrogram
        sr (int): the sampling rate of the audio file
        target_shape (tuple): target size (height * width). default (128, 128)
    """
    # Compute hop_length to get target size
    n_mels, width_target = target_shape
    # step between windows (smaller = better temporal resolution)
    hop_length = int(len(y) / (width_target - 1))

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec, hop_length=hop_length,
                             x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-spectrogram')
    plt.show()
