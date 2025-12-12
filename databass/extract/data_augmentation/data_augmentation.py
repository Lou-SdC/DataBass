## this module will do Data Augmentation on audio data
# now using librosa for I/O to avoid tensorflow-io dependency issues

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import tensorflow as tf
#TO DO
def augment_audio():
    #normalize audio files to have zero mean and unit variance
    WORKING_DIR = os.getenv("WORKING_DIR")
    #read file path from bass_list.csv located in WORKING_DIR/data/preprocessed
    #load the csv file
    bass_list = pd.read_csv(os.path.join(WORKING_DIR, "data", "preprocessed", "bass_list.csv"))
    #iterate through the file paths and normalize each audio file

    ### IMPLEMENT FUNCTIONS BELOW ###
    normalize_audio(bass_list)

#DONE
def normalize_audio_live_for_model(audio_tensor):
    """Normalize for ML input only - don't save to disk"""
    mean, variance = tf.nn.moments(audio_tensor, axes=[0])
    stddev = tf.sqrt(variance)
    return (audio_tensor - mean) / stddev

#DONE
def normalize_audio(df: pd.DataFrame):
    '''
    Docstring for normalize_audio

    :param df: The dataframe containing the path to audio files under file_path
    :type df: pd.DataFrame
    '''
    for index, row in df.iterrows():
        file_path = row['file_path']
        audio_wave, sample_rate = load_to_float32(file_path)

        # Peak normalize: scale so max absolute value is 1.0
        max_val = np.max(np.abs(audio_wave))
        if max_val == 0:
            continue
        normalized_audio = audio_wave / max_val

        # write back as int16 PCM
        sf.write(file_path + 'norm', normalized_audio, sample_rate, subtype='PCM_16')

#DONE
def fade_in_out (df: pd.DataFrame, fade_duration=0.5):
    '''
    Preprocess audio files by applying fade in and fade out effects.
    :param df: The dataframe containing the path to audio files under file_path
    :type df: pd.DataFrame
    :fade_duration: duration of the fade in/out in seconds
    :type fade_duration: float
    '''
    for index, row in df.iterrows():
        file_path = row['file_path']
        audio_wave, sample_rate = load_to_float32(file_path)
        num_fade_samples = int(fade_duration * sample_rate)
        if audio_wave.shape[0] <= 2 * num_fade_samples:
            continue

        # create fade in/out ramps
        fade_in_ramp = np.linspace(0.0, 1.0, num_fade_samples, dtype=np.float32)
        fade_out_ramp = np.linspace(1.0, 0.0, num_fade_samples, dtype=np.float32)

        faded = np.empty_like(audio_wave)
        faded[:num_fade_samples] = audio_wave[:num_fade_samples] * fade_in_ramp[:, None]
        faded[-num_fade_samples:] = audio_wave[-num_fade_samples:] * fade_out_ramp[:, None]
        faded[num_fade_samples:-num_fade_samples] = audio_wave[num_fade_samples:-num_fade_samples]

        sf.write(file_path + '_faded', faded, sample_rate, subtype='PCM_16')

#WIP
def frequency_masking(df: pd.DataFrame, freq_mask_param=15, num_masks=1):
    '''
    Docstring for frequency_masking

    :param df: The dataframe containing the path to audio files under file_path
    :type df: pd.DataFrame
    :freq_mask_param: maximum width of the frequency mask
    :type freq_mask_param: int
    :num_masks: number of frequency masks to apply
    :type num_masks: int
    '''
    for index, row in df.iterrows():
        file_path = row['file_path']
        audio_wave, sample_rate = load_to_float32(file_path)
        spectrogram = convert_to_spectrogram(audio_wave, sample_rate)

        masked = spectrogram.copy()
        num_bins = masked.shape[0]
        for _ in range(num_masks):
            f = np.random.randint(0, freq_mask_param + 1)
            f0 = np.random.randint(0, max(1, num_bins - f))
            masked[f0:f0 + f, :, :] = 0.0

        np.save(file_path + '_freqmask.npy', masked)

#DONE
def load_to_float32(file_path):
    """Load audio via librosa unnormaalized as float32 numpy array."""
    y, sr = librosa.load(file_path, sr=None, mono=False)
    return y.astype(np.float32), sr

#DONE
def convert_to_spectrogram(audio_wave, sample_rate, target_shape=(256, 256, 1)):
    """Compute Mel spectrogram with librosa and shape to (256, 256, 1)."""
    # collapse to mono for spectrogram generation
    if audio_wave.ndim == 2 and audio_wave.shape[1] > 1:
        audio_wave = np.mean(audio_wave, axis=1)
    elif audio_wave.ndim == 2:
        audio_wave = audio_wave[:, 0]

    mel = librosa.feature.melspectrogram(
        y=audio_wave,
        sr=sample_rate,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window='hann',
        n_mels=target_shape[0],
        power=2.0
    )

    # trim/pad time axis to target_shape[1]
    time_target = target_shape[1]
    if mel.shape[1] >= time_target:
        mel = mel[:, :time_target]
    else:
        pad_width = time_target - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')

    mel = mel.astype(np.float32)
    mel = np.expand_dims(mel, axis=-1)  # (256, time, 1)
    return mel
