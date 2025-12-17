import librosa
import pandas as pd
import numpy as np
import ast

import os
from databass.utils.create_table import create_table
from databass.utils.get_note_baseline import get_pic_frequency, get_note
from databass.preprocess.filter import frequencies_filter
from concurrent.futures import ProcessPoolExecutor
from alive_progress import alive_bar
from itertools import repeat


### BASELINE MODEL FUNCTIONS ###
### Attention ! These functions expect that raw audio files are in raw_data/ directory ###

def _predict_line(row, note_table):
    file_path = os.path.join(
        os.getenv('WORKING_DIR'),
        'raw_data',
        row.get('file_path'))
    print(f'Processing file: {file_path}')
    y, sr = librosa.load(file_path, sr=None, mono=True)
    pic_frequency, _, _ = get_pic_frequency(y, sr)
    note = get_note(pic_frequency, note_table) if pic_frequency is not None else ''
    pred_frequency = float(pic_frequency) if pic_frequency is not None else np.nan

    return {
        "file": row.get('file_path'),
        "target_note": row.get('note_name'),
        "target_frequency": librosa.note_to_hz(row.get('note_name')),
        "pred_frequency": pred_frequency,
        "pred_note": note.replace('♯', '#') if isinstance(note, str) else ""
    }

def predict(processed_file):
    df_files = pd.read_csv(processed_file)
    working_dir = os.getenv('WORKING_DIR')
    note_table = create_table()

    results = []
    rows = df_files.to_dict(orient='records')
    mw = os.cpu_count()
    print(f'Using {mw} workers for loading.')
    with ProcessPoolExecutor(max_workers=mw or 1) as executor:
        with alive_bar(len(rows), title='Predicting') as bar:
            for result in executor.map(_predict_line, rows, repeat(note_table)):
                results.append(result)
                bar()

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

def evaluate_pyin(prediction_file):
    df_predictions = pd.read_csv(prediction_file)

    total = len(df_predictions)
    correct = (df_predictions['target_note'] == df_predictions['pred_note']).sum()
    accuracy = correct / total if total > 0 else 0.0

    evaluation_text = f"Total samples: {total}\nCorrect predictions: {correct}\nAccuracy: {accuracy:.2%}"

    prediction_dir = os.path.dirname(prediction_file)
    evaluation_file = os.path.join(prediction_dir, 'pyin_evaluation.txt')
    with open(evaluation_file, 'w') as f:
        f.write(evaluation_text)

    return evaluation_text
