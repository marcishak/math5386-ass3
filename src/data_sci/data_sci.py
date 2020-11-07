from typing import List, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .neuralnetwrappers.nnModelInstance import ModelInstance as nnModelInstance
from .sklearnwrappers.skModelInstance import ModelInstance as skModelInstance


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

def _run_and_fit_nns(X_train:np.array, X_test:np.array, y_train:np.array, y_test:np.array, hidden_layers_size:List[Any]=[5], runs_per:int = 10):
    pass


def run_pipe(*args):
    pass