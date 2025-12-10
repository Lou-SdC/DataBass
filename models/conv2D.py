"""
Create a conv2D model with tensorflow Keras
"""

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import numpy as np

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


def preprocess_for_conv2D(X, y):

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # transforms note names in int
    y = np.array(y_encoded, dtype=np.int32)

    # Add the canal for Keras (1 = grey levels)
    X = np.expand_dims(X, axis=-1)

    # Split train/test/val
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

    return X_train, X_test, X_val, y_train, y_test, y_val
