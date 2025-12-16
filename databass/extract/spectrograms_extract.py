"""
Module that extracts the spectrograms based on the list file: exports one
.npy (numpy array) file for each sound file according to the bass_list.csv file
"""

import os
import librosa
import pandas as pd
import numpy as np

from preprocess.spectrograms import generate_mel_spectrogram as _generate_mel_spectrogram

def generate_mel_spectrogram(y, sr, target_shape = (256,256)):
    """
    Generate a Mel-spectrogram from an audio signal.

    Args:
        y (np.ndarray): Audio time series
        sr (int): Sampling rate of y
    Returns:
        mel_spectrogram (np.ndarray): The generated Mel-spectrogram
    """
    return _generate_mel_spectrogram(
        y,
        sr,
        normalize='minmax',
        target_shape=target_shape,
    )

def extract_spectrograms():

    # Récupérer WORKING_DIR
    WORKING_DIR = os.getenv("WORKING_DIR")

    #get the list file
    sample_list = pd.read_csv(WORKING_DIR + "/data/preprocessed/instruments_list_augmented.csv")

    # Dossier de sortie pour les spectrogrammes
    output_dir = WORKING_DIR + "/data/spectrograms"
    os.makedirs(output_dir, exist_ok=True)

    # Go through the dataframe
    for index, row in sample_list.iterrows():

        audio_path = row['file_path']

        # check that the file exists
        if not os.path.exists(audio_path):
            print(f"⚠️ Fichier introuvable : {audio_path}")
            continue

        # create the Mel-spectrogramme
        try:
            y, sr = librosa.load(audio_path)
            mel_spec = generate_mel_spectrogram(y, sr)
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {audio_path}: {e}")
            continue

        #Expand dims to make the spectrogram directly usable by the model (shape (128, 128, 1))
        mel_spec = np.expand_dims(mel_spec, axis=-1)

        # output folder for each note
        note_dir = os.path.join(output_dir, row['instrument'],row['note_name'].replace('#','sharp'))
        os.makedirs(note_dir, exist_ok=True)

        # output file name (.npy)
        output_filename = f"{row['fileID']}.npy"
        output_path = os.path.join(note_dir, output_filename)

        # Save the spectrogram .npy
        np.save(output_path, mel_spec)
        print(f"✅ Spectrogram saved: {output_path}")

if __name__ == "__main__":
    extract_spectrograms()
