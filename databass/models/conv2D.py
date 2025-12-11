"""
Create a conv2D model with tensorflow Keras
"""

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import os
import sys
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from preprocess.spectrograms import generate_mel_spectrogram
from tensorflow.keras.models import load_model as k_load_model
import pickle

import numpy as np

def create_model(input_shape=(128, 128, 1), num_classes=28, learning_rate=0.001):
    """
    Create a Conv2D model to classify the spectrograms.

    Args:
        input_shape (tuple): input shape (heigth, width, canals).
        num_classes (int): number of classes.
        learning_rate (float): learning rate of the Adam optimizer

    Returns:
        tf.keras.Model: compiled model.
    """
    model = models.Sequential([
        # Couches de convolution
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Aplatissement et couches denses
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compilation
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def preprocess_for_conv2D(X, y):
    """Preprocess the data for model training : get X in the right
    form for feeding the model and encode the y labels in int values

    Args:
        X (np.ndarray): array of spectrograms
        y (list): the list of targets (list of notes, ex: 'E1')

    Returns:
        X_train, X_test, X_val: np.ndarrays inputs ready
            for model training and evaluation
        y_train, y_test, y_val: np.ndarrays targets ready
            for model training and evaluation
        le: LabelEncoder used to encode the targets values,
            will be useful to decode it later
    """

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # transforms note names in int
    y = np.array(y_encoded, dtype=np.int32)

    # Add the canal for Keras (1 = grey levels)
    X = np.expand_dims(X, axis=-1)

    # Split train/test/val
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

    return X_train, X_test, X_val, y_train, y_test, y_val, le


def preprocess_for_predict_conv2D(X):
    """
    Prepare a spectrogram for prediction using a conv2D model generated
    with conv2D.create_model()

    Args:
        X (np.ndarray): the spectrogram of the note we want to identify

    Returns:
        X (np.ndarray): a spectrogram with the right shape for a prediction
            by the model
    """
    X = np.expand_dims(X, axis=-1)
    X = np.expand_dims(X, axis=0)

    return X


def conv2D_predict_note(X, model, le):
    """A function to get the predicted note for a given spectrogram

    Args:
        X (np.ndarray): The input spectrogram
        model (keras model): the model
        le (LabelEncoder): the encoder used to encode the targets

    Returns:
        predicted_note (str): the note that was predicted by the model
    """
    result = model.predict(X)

    predicted_class = np.argmax(result, axis=1)

    # Get the note from the class
    predicted_note = le.inverse_transform(predicted_class)

    return predicted_note[0]

def predict(signal, sr, model, le):
    spec = generate_mel_spectrogram(signal, sr, normalize="minmax")
    processed = preprocess_for_predict_conv2D(spec)

    return conv2D_predict_note(processed, model, le)

def load_model():
    """Load the conv2D model and the label encoder from disk
    Returns:
        model (keras model): the loaded model
        le (LabelEncoder): the loaded label encoder
    """
    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PARENT_DIR = os.path.dirname(PARENT_DIR)
    MODEL_PATH = os.path.join(PARENT_DIR, 'data', 'models', 'conv2D_model.keras')
    model = k_load_model(MODEL_PATH)
    print("Model loaded from", MODEL_PATH)
    # load label encoder from label_encoder.pkl file
    le_path = os.path.join(PARENT_DIR, 'data', 'models', 'conv2D_label_encoder.pkl')

    with open(le_path, 'rb') as f:
        le = pickle.load(f)

    return model, le
