"""
module to filter the signal and plot the results
"""

import numpy as np
import matplotlib.pyplot as plt


def frequencies_filter(y, sr, lower_freq=35, higher_freq=405):
    """
    Filters an audio signal to keep only frequencies within a specified band.

    Args:
        y (np.ndarray): Time-domain audio signal.
        sr (int): Sampling rate (Hz).
        lower_freq (float): Lower cutoff frequency (Hz). Default: 35 Hz.
        higher_freq (float): Upper cutoff frequency (Hz). Default: 405 Hz.

    Returns:
        np.ndarray: Filtered time-domain audio signal.
    """
    # Compute the FFT of the signal
    n = len(y)
    yf = np.fft.fft(y)

    # Compute the corresponding frequencies (only positive half)
    freq_bins = np.fft.fftfreq(n, d=1/sr)

    # Create a boolean mask for frequencies to keep
    # We only need to consider positive frequencies for real signals
    mask = (np.abs(freq_bins) >= lower_freq) & (np.abs(freq_bins) <= higher_freq)

    # Apply the mask to the FFT result
    yf_filtered = yf.copy()
    yf_filtered[~mask] = 0

    # Convert back to the time domain using the inverse FFT
    y_filtered = np.real(np.fft.ifft(yf_filtered))

    return y_filtered

def plot_filtered_vs_original(y, y_filtered):
    # Plot the original and filtered signals
    plt.figure(figsize=(12, 6))
    plt.subplot(4, 1, 1)
    plt.plot(y, label='Original')
    plt.title('Original Signal')
    plt.subplot(4, 1, 2)
    plt.plot(y_filtered, label='Filtered', color='red')
    plt.title('Filtered Signal')
    plt.tight_layout()
    plt.figure(figsize=(12, 6))
    plt.subplot(4, 1, 3)
    plt.plot(y[10000:15000], label='Original')
    plt.title('Original Signal')
    plt.subplot(4, 1, 4)
    plt.plot(y_filtered[10000:15000], label='Filtered', color='red')
    plt.title('Filtered Signal')
    plt.tight_layout()
    plt.show()
