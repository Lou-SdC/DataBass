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
    # find current execution path
    WORKING_DIR = os.getcwd()
    print("Working dir:", WORKING_DIR)
    PARENT_DIR = os.path.dirname(WORKING_DIR)
    print("Parent dir:", PARENT_DIR)
    MODEL_PATH = os.path.join(PARENT_DIR, 'data', 'models', 'RandForClassifier.pkl')
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    return model

def select_starting_point(X, length_before=1000, length_after=5000):
    """
    Truncate file to just before and after the start of the note.
    """
    start_ind = 0
    for i in range(len(X)):
        if X[i] >= 0.9 * X.max():
            start_ind = i
            break

    X_select = X[start_ind-length_before:start_ind+length_after].copy()
    return X_select
