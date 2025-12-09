### this module intialize a keras Dense model that input audio files extracted by librosa
### of a bass guitar and outputs the individual notes played in these files.
import tensorflow as tf
from keras import layers, models

def create_dense_model(input_shape, num_classes):
    """
    Creates a simple Dense neural network model.

    Args:
        input_shape (tuple): Shape of the input data (excluding batch size).
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: Compiled Dense model.
    """

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
