from typing import List, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from .neuralnetwrappers.nnModelInstance import ModelInstance as nnModelInstance
from .sklearnwrappers.skModelInstance import ModelInstance as skModelInstance
from .sklearnwrappers.skModelInstance import summarise_model_instances


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


def _fit_and_test_sk(X_train:np.array, X_test:np.array, y_train:np.array, y_test:np.array, model:str, runs_per:int = 10, random_state=69):
    model_list = []
    for _ in range(runs_per):
        sk_model = skModelInstance(X_train, X_test, y_train, y_test, eval(model+"()"), [*range(X_train.shape[1])], 1000)
        sk_model.fit_predict_model()
        sk_model.summarise_model_instance()
        model_list.append(sk_model)
    summarise_model_instances(model_list).to_csv("data/output/"+model+"-sklearnsummary.csv")


def run_pipe(X_train, X_test, y_train, y_test, models):
    _fit_and_test_sk(X_train, X_test, y_train, y_test, models)