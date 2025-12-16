## this module will do Data Augmentation on audio data
# now using librosa for I/O to avoid tensorflow-io dependency issues

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import tensorflow as tf

import sys
from pathlib import Path

# import using the package root so it works when run as a module
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from utils import bass_notes
from utils import guitar_notes

from preprocess.spectrograms import generate_mel_spectrogram, plot_mel_spectrogram



#TO DO
def augment_audio():

    # Récupérer WORKING_DIR
    WORKING_DIR = os.getenv("WORKING_DIR")

    #get the list file
    sample_list = pd.read_csv(WORKING_DIR + "/data/preprocessed/instruments_list.csv")
    new_sample_list = sample_list.copy()
    # Dossier de sortie pour les spectrogrammes
    output_dir = WORKING_DIR + "/data/spectrograms"
    os.makedirs(output_dir, exist_ok=True)

    # Go through the dataframe
    for index, row in sample_list.iterrows():

        audio_path = row['file_path']

        y, sr = librosa.load(audio_path)

        fade_in_out(row, y, sr, new_sample_list)
        add_noise(row, y, sr, new_sample_list)
        dim_audio(row, y, sr, new_sample_list)
        shift_audio(row, y, sr, new_sample_list)

        normalize_audio(row, y, sr, new_sample_list)



    for index, row in new_sample_list.iterrows():
        y, sr = librosa.load(row['file_path'])
        frequency_masking(row, y, sr)

    #save new sample list to WORKING_DIR/data/preprocessed/instruments_list_augmented.csv
    new_sample_list.to_csv(WORKING_DIR + "/data/preprocessed/instruments_list_augmented.csv", index=False)


def normalize_audio_live_for_model(audio_tensor):
    """Normalize for ML input only - don't save to disk"""
    mean, variance = tf.nn.moments(audio_tensor, axes=[0])
    stddev = tf.sqrt(variance)
    return (audio_tensor - mean) / stddev

def normalize_audio(row, audio_wave: np.ndarray,sr, new_sample_list: pd.DataFrame):

    # Peak normalize: scale so max absolute value is 1.0
    max_val = np.max(np.abs(audio_wave))
    normalized_audio = audio_wave / max_val

    append_augmentation_to_list(row, normalized_audio, sr, "normalized", "norm_", new_sample_list)

def fade_in_out(row, audio_wave , sr , new_sample_list: pd.DataFrame, fade_duration=0.5):
    num_fade_samples = int(fade_duration * sr)

    fade_in_ramp = np.linspace(0.0, 1.0, num_fade_samples, dtype=np.float32)
    fade_out_ramp = np.linspace(1.0, 0.0, num_fade_samples, dtype=np.float32)

    faded = np.empty_like(audio_wave)
    # reshape ramp for broadcasting: mono is (n,), stereo is (n, 2), etc
    if audio_wave.ndim == 1:
        faded[:num_fade_samples] = audio_wave[:num_fade_samples] * fade_in_ramp
        faded[-num_fade_samples:] = audio_wave[-num_fade_samples:] * fade_out_ramp
    else:
        # multi-channel: reshape ramp to (num_fade_samples, 1) for broadcasting
        faded[:num_fade_samples] = audio_wave[:num_fade_samples] * fade_in_ramp[:, None]
        faded[-num_fade_samples:] = audio_wave[-num_fade_samples:] * fade_out_ramp[:, None]
    faded[num_fade_samples:-num_fade_samples] = audio_wave[num_fade_samples:-num_fade_samples]

    append_augmentation_to_list(row, faded, sr, "fade_in_out", "fadeinout_", new_sample_list)

def add_noise(row , audio_wave, sr, new_sample_list: pd.DataFrame, noise_factor=0.005):
    """Add random noise to an audio signal."""
    noise = np.random.randn(len(audio_wave))
    augmented_audio = audio_wave + noise_factor * noise

    noisy_audio = augmented_audio.astype(type(audio_wave[0]))

    append_augmentation_to_list(row, noisy_audio, sr, "noisy", "noisy_", new_sample_list)

def frequency_masking(row, audio_wave, sr, freq_mask_param=15, num_masks=1):

    output_dir = os.path.join(PROJECT_ROOT, "../data/spectrograms")

    spectrogram = generate_mel_spectrogram(
        audio_wave,
        sr,
        normalize='minmax',
        target_shape=(256,256)
    )

    masked = spectrogram.copy()
    num_bins = masked.shape[0]

    mask_width = max(1, freq_mask_param + np.random.randint(-5, 6))

    for _ in range(num_masks):
        f = np.random.randint(1, mask_width + 1)
        f0 = np.random.randint(0, max(1, num_bins - f))
        masked[f0:f0 + f, :] = 0.0

    # Expand dims to make the spectrogram directly usable by the model (shape (256, 256, 1))
    mel_spec = np.expand_dims(masked, axis=-1)

    # output folder for each note
    note_dir = os.path.join(output_dir, row['note_name'])
    os.makedirs(note_dir, exist_ok=True)

    # output file name (.npy)
    output_filename = f"masked_{row['fileID']}.npy"
    output_path = os.path.join(note_dir, output_filename)

    # Save the spectrogram .npy
    np.save(output_path, mel_spec)

def dim_audio(row,audio_wave: np.ndarray, sr, instrument_list: pd.DataFrame, dim_factor=0.5):
    """reduce the amplitude as to simulate less audible audio files"""

    dimmed_audio = audio_wave * (dim_factor * np.random.rand())  # reduce amplitude by dim_factor and random

    append_augmentation_to_list(row, dimmed_audio, sr, "dimmed", "dimmed_", instrument_list)

def shift_audio(row, audio_wave: np.ndarray, sr, new_sample_list: pd.DataFrame, shift_max=0.2):

    shift = np.random.randint(int(sr * -shift_max), int(sr * shift_max))
    shifted_audio = np.roll(audio_wave, shift)

    append_augmentation_to_list(row, shifted_audio, sr, "shifted", "shifted_", new_sample_list)

def append_augmentation_to_list(row, audio_wave, sr, directory, prefix, instrument_list):

    file_name = os.path.basename(row['file_path'])

    norm_dir = os.path.join(PROJECT_ROOT, "../data", directory)
    if not os.path.exists(norm_dir):
        os.makedirs(norm_dir)

    new_file_path = os.path.join(norm_dir, prefix + file_name)
    sf.write(new_file_path, audio_wave, sr)
    print('file created at : ', new_file_path)

    new_row = row.copy()
    new_row['file_path'] = new_file_path
    instrument_list = pd.concat([instrument_list, pd.DataFrame([new_row])], ignore_index=True)

if __name__ == "__main__":
    augment_audio()
