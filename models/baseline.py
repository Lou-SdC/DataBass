import librosa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sys
import os
from dotenv import load_dotenv
from utils.create_table import create_table
from utils.get_note_baseline import get_pic_frequency, get_note, plot_fft
from preprocess.filter import frequencies_filter
import glob

def predict(processed_file):
    df_files = pd.read_csv(processed_file)
    working_dir = os.getenv('WORKING_DIR')
    note_table = create_table()
    results = []

    print(working_dir)
    for _, row in df_files.iterrows():
        audio_file = os.path.join(working_dir, 'raw_data', row['file_path'])
        print(f"Processing file: {audio_file}")
        y, sr = librosa.load(audio_file, sr=None)

        # Filter the signal to keep only bass guitar frequencies
        y = frequencies_filter(y, sr, lower_freq=35, higher_freq=405)

        # Using only the max FFT peak to estimate frequency
        pic_frequency, magnitude, frequencies = get_pic_frequency(y, sr)
        note = get_note(pic_frequency, note_table)

        results.append({
            "file": audio_file,
            "target_note": row['note_name'],
            "target_frequency": librosa.note_to_hz(row['note_name']),
            "pred_frequency": float(pic_frequency) if pic_frequency is not None else np.nan,
            "pred_note": note
        })

    # build output directory next to project data folder: <project_root>/data/baseline
    prediction_dir = os.path.join(working_dir, 'data', 'baseline')
    os.makedirs(prediction_dir, exist_ok=True)
    prediction_file = os.path.join(prediction_dir, 'notes.csv')
    pd.DataFrame(results).to_csv(prediction_file, index=False)

    return prediction_file

def evaluate(prediction_file):
    df_predictions = pd.read_csv(prediction_file)

    total = len(df_predictions)
    correct = (df_predictions['target_note'] == df_predictions['pred_note']).sum()
    accuracy = correct / total if total > 0 else 0.0

    evaluation_text = f"Total samples: {total}\nCorrect predictions: {correct}\nAccuracy: {accuracy:.2%}"

    prediction_dir = os.path.dirname(prediction_file)
    evaluation_file = os.path.join(prediction_dir, 'evaluation.txt')
    with open(evaluation_file, 'w') as f:
        f.write(evaluation_text)

    return evaluation_text
