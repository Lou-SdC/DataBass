# Preprocess to load audio files using librosa
# This is done to ensure faster processing

import librosa
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
from alive_progress import alive_bar
import numpy as np


def load_audio_file(entry):
    """
    Load an audio file using librosa.

    Parameters:
    - file_path (str): Path to the audio file.
    - sr (int or None): Sample rate for loading. If None, uses the file's original sample rate.

    Returns:
    - y (np.ndarray): Audio time series.
    - sr (int): Sample rate of y.
    """
    audio_file = entry["audio_file"]
    target_note = entry["target_note"]
    target_frequency = entry["target_frequency"]
    audio_time_serie, sr = librosa.load(audio_file, sr=None)
    return {
        "file": audio_file,
        "target_note": target_note,
        "target_frequency": target_frequency,
        "audio_time_serie": np.asarray(audio_time_serie, dtype=np.float32).tolist(),
        "sample_rate": sr
    }

def load_audio_files(file_paths):
    """"
    Load multiple audio files using librosa.
    Save the results in a CSV file.
    Parameters:
    - file_paths (str): Path to the CSV file containing audio file paths.

    Returns:
    - output_file (str): Path to the CSV file containing loaded audio data.
    """

    df_files = pd.read_csv(file_paths)
    working_dir = os.getenv('WORKING_DIR')
    tasks = []
    for _, row in df_files.iterrows():
        audio_file = os.path.join(working_dir, 'raw_data', row['file_path'])
        tasks.append({
            "audio_file": audio_file,
            "target_note": row['note_name'],
            "target_frequency": librosa.note_to_hz(row['note_name'])
        })
    mw = os.cpu_count()
    print(f'Using {mw} workers for loading.')
    results = []
    with ProcessPoolExecutor(max_workers=mw or 1) as executor, alive_bar(len(tasks)) as bar:
        for result in executor.map(load_audio_file, tasks):
            results.append(result)
            bar()
    df_results = pd.DataFrame(results)
    output_file = os.path.join(working_dir, 'data', 'preprocessed', 'librosa_loaded_audio.csv')
    df_results.to_csv(output_file, index=False)
    return output_file
