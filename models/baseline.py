import librosa
import pandas as pd
import numpy as np

import os
from utils.create_table import create_table
from utils.get_note_baseline import get_pic_frequency, get_note
from preprocess.filter import frequencies_filter
from concurrent.futures import ProcessPoolExecutor
from alive_progress import alive_bar

_NOTE_TABLE = None

def _init_worker(note_table):
    global _NOTE_TABLE
    _NOTE_TABLE = note_table

def _process_file_entry(entry):
    global _NOTE_TABLE
    audio_file = entry["audio_file"]
    target_note = entry["target_note"]
    target_frequency = entry["target_frequency"]
    try:
        y, sr = librosa.load(audio_file, sr=None)
        y = frequencies_filter(y, sr, lower_freq=35, higher_freq=405)
        pic_frequency, _, _ = get_pic_frequency(y, sr)
        note = get_note(pic_frequency, _NOTE_TABLE)
        pred_frequency = float(pic_frequency) if pic_frequency is not None else np.nan
    except Exception:
        pred_frequency = np.nan
        note = None
    return {
        "file": audio_file,
        "target_note": target_note,
        "target_frequency": target_frequency,
        "pred_frequency": pred_frequency,
        "pred_note": note.replace('♯', '#')
    }

def predict(processed_file):
    df_files = pd.read_csv(processed_file)
    working_dir = os.getenv('WORKING_DIR')
    print(working_dir)
    note_table = create_table()
    tasks = []
    for _, row in df_files.iterrows():
        audio_file = os.path.join(working_dir, 'raw_data', row['file_path'])
        print(f"Processing file: {audio_file}")
        tasks.append({
            "audio_file": audio_file,
            "target_note": row['note_name'],
            "target_frequency": librosa.note_to_hz(row['note_name'])
        })
    if tasks:
        mw = os.cpu_count()
        print(f'Using {mw} workers for processing.')
        results = []
        with ProcessPoolExecutor(max_workers=mw or 1, initializer=_init_worker, initargs=(note_table,)) as executor, alive_bar(len(tasks)) as bar:
            for result in executor.map(_process_file_entry, tasks):
                results.append(result)
                bar()
    else:
        results = []
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
