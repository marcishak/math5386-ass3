from typing import List, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from .neuralnetwrappers.nnModelInstance import ModelInstance as nnModelInstance
from .sklearnwrappers.skModelInstance import ModelInstance as skModelInstance
from .sklearnwrappers.skModelInstance import (
    summarise_model_instances as sk_summarise_model_instances,
)
from .neuralnetwrappers.nnModelInstance import (
    summarise_model_instances as nn_summarise_model_instances,
)


def _build_layers(hidden_layers):
    """
    wish we had lazy eval here
    """
    layers = [
        [
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
    return list(zip(layers, hidden_layers))


def _fit_and_test_nn(
    X_train: np.array,
    X_test: np.array,
    y_train: np.array,
    y_test: np.array,
    model_type: str,
    opt_funcs: List[Any] = [keras.optimizers.Adam(), keras.optimizers.SGD()],
    hidden_layers_sizes: List[Any] = [5],
    runs_per: int = 10,
):
    model_list = []
    hidden_layers_sizes_out = []
    layer_lengths = []
    if "Classi" in model_type:
        output_act = "sigmoid"
        loss_func = keras.losses.MeanSquaredError()
        metrics = ["AUC"]
    elif "Regress" in model_type:
        output_act = None
        loss_func = keras.losses.MeanSquaredError()
        metrics = ["MSE"]
    else:
        ValueError("Invalid Model Type")
    built_layers = _build_layers(hidden_layers_sizes)
    print(opt_funcs)
    for opt_func in opt_funcs:
        print(opt_func)
        for layer_group, hidden_layer_size in built_layers:
            for layers in layer_group:
                for _ in range(runs_per):
                    nn_model = nnModelInstance(
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        layers,
                        output_activation=output_act,
                        loss_func=loss_func,
                        opt_func=opt_func,
                        metrics=metrics,
                    )
                    nn_model.fit_predict_model(
                        "validation",
                        epochs=300,
                        batch_size=round((X_train.shape[0]) / 10),
                    )
                    nn_model.summarise_model_instance()
                    model_list.append(nn_model)
                    hidden_layers_sizes_out.append(hidden_layer_size)
                    layer_lengths.append(len(layers))
    nn_summarise_model_instances(
        model_list, hidden_layers_sizes_out, layer_lengths
    ).to_csv("data/output/" + model_type + "-nnsummary.csv", index=False)


def _fit_and_test_sk(
    X_train: np.array,
    X_test: np.array,
    y_train: np.array,
    y_test: np.array,
    model: str,
    runs_per: int = 10,
):
    model_list = []
    for _ in range(runs_per):
        sk_model = skModelInstance(
            X_train,
            X_test,
            y_train,
            y_test,
            eval(model + "()"),
            [*range(X_train.shape[1])],
            1000,
        )
        sk_model.fit_predict_model()
        sk_model.summarise_model_instance()
        model_list.append(sk_model)
    sk_summarise_model_instances(model_list).to_csv(
        "data/output/" + model + "-sklearnsummary.csv", index=False
    )


def sk_run_pipe(X_train, X_test, y_train, y_test, models):
    _fit_and_test_sk(X_train, X_test, y_train, y_test, models)


def nn_run_pipe(X_train, X_test, y_train, y_test, models):
    _fit_and_test_nn(X_train, X_test, y_train, y_test, models)
