"""
Create a conv2D model with tensorflow Keras
"""

import os
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras.utils import to_categorical

import pickle

import numpy as np

def create_model(params: dict = {'n_estimators': 100, 'max_depth': None, 'max_leaf_nodes': None}):
    """
    Create a Random Forest Classifier model,

    expects a dictionnary as params with :
    - n_estimators (int): number of trees in the forest default 100
    - max_depth (int): maximum depth of the trees default None
    - max_leaf_nodes (int): maximum number of leaf nodes in the trees default None
    """

    params = params

    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_leaf_nodes=params['max_leaf_nodes'],
        max_depth=params['max_depth'],
        )

    return model


def preprocess(X, y):
    """Preprocess the data for model training : get X in the right
    form : Wave forms of audio file of np.ndarray type
    for feeding the model and encode the y labels in int values.

    Args:
        X (np.ndarray): a list of waveforms in np.ndarray form
        y (list): the list of targets (list of notes, ex: 'E1')

    Returns:
        X_train, X_test, X_val: np.ndarrays of float32 inputs ready
            for model training and evaluation
        y_train, y_test, y_val: np.ndarrays targets ready
            for model training and evaluation
        le: LabelEncoder used to encode the targets values,
            will be useful to decode it later
    """

    le = LabelEncoder()
    y_int = le.fit_transform(y)  # integers 0..n_classes-1
    y = to_categorical(y_int)

    # Apply the function to each element in X
    X = [select_starting_point(x) for x in X]
    # Stack the selected portions into a single array
    X = np.stack(X).astype(np.float32)


    # Split train/test/val
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

    return X_train, X_test, X_val, y_train, y_test, y_val, le


def rand_forest_predict_note(X, model):
    """A function to get the predicted note for a given audio file
    in np.ndarray(float32) form.

    Args:
        X (np.ndarray): The input spectrogram
        model (sklearn.ensemble.RandomForestClassifier): the model
        le (LabelEncoder): the encoder used to encode the targets

    Returns:
        predicted_note (str): the note that was predicted by the model
    """
    result = model.predict(X)

    return result[0]

def predict(single_file, model, sr = 22050):
    """
    Returns the note of an audio file called with this function based on
    the model.
    """
    X , _ = librosa.load(single_file, sr=sr,dtype=np.float32)
    X = select_starting_point(X)
    X = np.stack([X]).astype(np.float32)

    return rand_forest_predict_note(X, model)

def load_model():
    # Find the repository root (where databass package is located)
    current_file = os.path.abspath(__file__)
    # Go up from databass/models/rand_forest.py to DataBass/
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    print("Repository root:", REPO_ROOT)
    MODEL_PATH = os.path.join(REPO_ROOT, 'data', 'models', 'RandForClassifier.pkl')
    print("Model path:", MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"❌ RandomForest model not found at: {MODEL_PATH}\n"
            f"   Please ensure the model file exists in the data/models/ directory."
        )

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    return model

def evaluate(X_test, y_test):
    model = load_model()

    result = model.score(X_test, y_test)

    return result

def select_starting_point(X, length_before=1000, length_after=5000):
    """
    Truncate file to just before and after the start of the note.
    If the audio is too short, pad it with zeros to reach the expected length.

    Args:
        X: audio waveform
        length_before: samples before the peak (default: 1000)
        length_after: samples after the peak (default: 5000)

    Returns:
        X_select: audio segment of exactly (length_before + length_after) samples
    """
    expected_length = length_before + length_after

    # Handle empty or very short arrays
    if len(X) == 0:
        return np.zeros(expected_length, dtype=np.float32)

    # Find the peak (start of note)
    start_ind = 0
    max_val = X.max()
    if max_val > 0:  # Avoid issues with silent audio
        for i in range(len(X)):
            if X[i] >= 0.9 * max_val:
                start_ind = i
                break

    # Extract the segment
    start_slice = max(0, start_ind - length_before)
    end_slice = min(len(X), start_ind + length_after)
    X_select = X[start_slice:end_slice].copy()

    # Pad with zeros if too short
    if len(X_select) < expected_length:
        padding_needed = expected_length - len(X_select)
        # Pad at the end
        X_select = np.pad(X_select, (0, padding_needed), mode='constant', constant_values=0)
    # Truncate if too long (shouldn't happen with the slicing above, but just in case)
    elif len(X_select) > expected_length:
        X_select = X_select[:expected_length]

    return X_select.astype(np.float32)
