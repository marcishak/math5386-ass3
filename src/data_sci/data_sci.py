from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras

from neuralnetwrappers.nnModelInstance import ModelInstance as nnModelInstance 


def _build_layers(hidden_layers):
    """
    wish we had lazy eval here
    """
    layers = [
        [
            [
                keras.layers.Dense(
                    x, kernel_initializer="random_normal", activation="relu"
                )
            ],
            [
                keras.layers.Dense(
                    x, kernel_initializer="random_normal", activation="relu"
                ),
                keras.layers.Dense(
                    x, kernel_initializer="random_normal", activation="relu"
                ),
            ],
            [
                keras.layers.Dense(
                    x, kernel_initializer="random_normal", activation="relu"
                ),
                keras.layers.Dense(
                    x, kernel_initializer="random_normal", activation="relu"
                ),
                keras.layers.Dense(
                    x, kernel_initializer="random_normal", activation="relu"
                ),
                keras.layers.Dense(
                    x, kernel_initializer="random_normal", activation="relu"
                ),
                keras.layers.Dense(
                    x, kernel_initializer="random_normal", activation="relu"
                ),
            ],
        ]
        for x in hidden_layers
    ]
    return zip(layers, hidden_layers)

def run_and_fit_nns(X_train:np.array, X_test:np.array, y_train:np.array, y_test:np.array, hidden_layers_size:List[5], runs_per:int = 10):
