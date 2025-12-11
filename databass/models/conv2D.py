"""
Create a conv2D model with tensorflow Keras
"""

from keras import layers, models, optimizers
from keras.callbacks import EarlyStopping
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from databass.preprocess.spectrograms import generate_mel_spectrogram
from keras.models import load_model as k_load_model
import pickle

import numpy as np


def _pad_or_crop_spec(spec: np.ndarray, target_shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Ensure the mel-spectrogram has the expected (freq, time) size.

    Strategy (robust for short notes):
    - Center-crop if dimension is larger than target
    - Symmetric zero-pad if smaller (keeps note content centered)
    """
    target_f, target_t = target_shape
    f, t = spec.shape

    # Adjust frequency axis (rows)
    if f < target_f:
        pad_top = (target_f - f) // 2
        pad_bottom = target_f - f - pad_top
        spec = np.pad(spec, ((pad_top, pad_bottom), (0, 0)), mode="constant")
    elif f > target_f:
        start = (f - target_f) // 2
        spec = spec[start:start + target_f, :]

    # Adjust time axis (columns)
    if t < target_t:
        pad_left = (target_t - t) // 2
        pad_right = target_t - t - pad_left
        spec = np.pad(spec, ((0, 0), (pad_left, pad_right)), mode="constant")
    elif t > target_t:
        start = (t - target_t) // 2
        spec = spec[:, start:start + target_t]

    return spec.astype(np.float32)

def create_model(input_shape=(128, 128, 1), num_classes=41, learning_rate=0.001):
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


def preprocess_for_conv2D(X, y, load_encoder=False):
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
    if load_encoder:
        # Find the repository root (where databass package is located)
        current_file = os.path.abspath(__file__)
        # Go up from databass/models/conv2D.py to DataBass/
        REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        print("Repository root:", REPO_ROOT)
        ENCODER_PATH = os.path.join(REPO_ROOT, 'data', 'models', 'conv2D_label_encoder.pkl')
        # Load the encoder from a pickle file if specified
        with open(ENCODER_PATH, 'rb') as f:
            le = pickle.load(f)
        y_encoded = le.transform(y)  # transforms note names in int
        y = np.array(y_encoded, dtype=np.int32)

        if not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(
                f"❌ Conv2D encoder not found at: {ENCODER_PATH}\n"
                f"   Please ensure the encoder file exists in the data/models/ directory."
            )

    # Initialize and fit a new encoder
    else:
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
    # Force fixed (freq, time) shape for the model
    X = _pad_or_crop_spec(X, target_shape=(128, 128))

    # Add channel and batch dimensions: (1, 128, 128, 1)
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
    # Find the repository root (where databass package is located)
    current_file = os.path.abspath(__file__)
    # Go up from databass/models/conv2D.py to DataBass/
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    print("Repository root:", REPO_ROOT)
    MODEL_PATH = os.path.join(REPO_ROOT, 'data', 'models', 'conv2D_model.keras')
    print("Model path:", MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"❌ Conv2D model not found at: {MODEL_PATH}\n"
            f"   Please ensure the model file exists in the data/models/ directory."
        )

    model = k_load_model(MODEL_PATH)

    # load label encoder from label_encoder.pkl file
    le_path = os.path.join(REPO_ROOT, 'data', 'models', 'conv2D_label_encoder.pkl')

    if not os.path.exists(le_path):
        raise FileNotFoundError(
            f"❌ Label encoder not found at: {le_path}\n"
            f"   Please ensure the label encoder file exists in the data/models/ directory."
        )

    with open(le_path, 'rb') as f:
        le = pickle.load(f)

    return model, le
