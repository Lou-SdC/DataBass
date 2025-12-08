"""
Module to get the pic_frequency of an audio sample to map it to the corresponding note
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_pic_frequency(y, sr):
    """
    Get the pic_frequency of a preprocessed audio sample

    Args:
        y (np array): a np array obtained with librosa.load()
        sr (int): the sampling rate of the audio file

    Returns:
        pic_frequency (float): the pic_frequency of the sample
        magnitude (np array): the result of the FFT transformation
        frequencies (np array): the frequency range
    """

    start_ind = 0
    i = 0
    while start_ind==0:
        if y[i] >= 0.9 * y.max():
            start_ind = i
        else:
            i += 1

    y_select = y[start_ind:start_ind+5000].copy()

    fft_result = np.fft.fft(y_select, sr)

    # take magnitude for analysis
    magnitude = np.abs(fft_result)

    # take only positive spectrum
    n = len(y)
    frequencies = np.linspace(0, sr, n)[:n//2]
    magnitude = magnitude[:n//2]

    pic_index = np.argmax(magnitude)
    pic_frequency = frequencies[pic_index]

    return pic_frequency, magnitude, frequencies


def plot_fft(magnitude, frequencies):
    """a function that plots the magnitude for each frequency range

    Args:
        magnitude (np array): the result of the FFT transformation
        frequencies (np array): the frequency range
    """
    plt.figure(figsize=(10, 4))
    plt.plot(frequencies, magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency spectrum')
    plt.xlim((35,400))
    plt.grid()
    plt.show()


def get_note(pic_frequency, df_notes):
    """a function to get the closest note in our table

    Args:
        pic_frequency (float): the pic frequency previously computed
        df_notes (pd dataframe): the table that maps frequencies to notes

    Returns:
        string: the note
    """
    #Get the closest note in the table
    closest_note = df_notes.iloc[(df_notes['fréquence (Hz)'] - pic_frequency).abs().argsort()[:1]]
    return closest_note['note'].values[0]
