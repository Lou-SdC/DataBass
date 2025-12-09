import librosa
import pandas as pd
import numpy as np
import ast

import os
from utils.create_table import create_table
from utils.get_note_baseline import get_pic_frequency, get_note
from preprocess.filter import frequencies_filter
from concurrent.futures import ProcessPoolExecutor
from alive_progress import alive_bar
from itertools import repeat

def _predict_line(row, note_table):
    ats = row.get('audio_time_serie', None)
    if ats is None or (isinstance(ats, float) and np.isnan(ats)) or ats == '':
        y = np.array([], dtype=np.float32)
    elif isinstance(ats, (list, np.ndarray)):
        y = np.asarray(ats, dtype=np.float32)
    elif isinstance(ats, str):
        try:
            parsed = ast.literal_eval(ats)
            y = np.asarray(parsed, dtype=np.float32)
        except Exception:
            # fallback: try parsing numeric values separated by commas or spaces
            try:
                y = np.fromstring(ats, sep=',', dtype=np.float32)
                if y.size == 0:
                    y = np.fromstring(ats, sep=' ', dtype=np.float32)
            except Exception:
                y = np.array([], dtype=np.float32)
    else:
        try:
            y = np.asarray(ats, dtype=np.float32)
        except Exception:
            y = np.array([], dtype=np.float32)

    sr = row.get('sample_rate', None)
    if isinstance(sr, str):
        # try to coerce common numeric string formats to int
        if sr.isdigit():
            sr = int(sr)
        else:
            try:
                sr = int(float(sr))
            except Exception:
                sr = None

    # If no valid audio or sample rate, return NaN/empty prediction to avoid crashes
    if y.size == 0 or sr is None or (isinstance(sr, float) and np.isnan(sr)):
        return {
            "file": row.get('file'),
            "target_note": row.get('target_note'),
            "target_frequency": row.get('target_frequency'),
            "pred_frequency": np.nan,
            "pred_note": ""
        }

    y = frequencies_filter(y, sr, lower_freq=35, higher_freq=405)
    pic_frequency, _, _ = get_pic_frequency(y, sr)
    note = get_note(pic_frequency, note_table) if pic_frequency is not None else ''
    pred_frequency = float(pic_frequency) if pic_frequency is not None else np.nan

    return {
        "file": row.get('file'),
        "target_note": row.get('target_note'),
        "target_frequency": row.get('target_frequency'),
        "pred_frequency": pred_frequency,
        "pred_note": note.replace('♯', '#') if isinstance(note, str) else ""
    }

def predict(lodaed_df_file):
    df_files = pd.read_csv(lodaed_df_file)
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
