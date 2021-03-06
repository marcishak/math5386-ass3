import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_curve,
    roc_auc_score,
    f1_score,
    r2_score,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from datetime import datetime
import os
from .nnModelWrapper import ModelWrapper
from typing import List


class ModelInstance:
    """
    Wrapper for each model instance
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        layers,
        hidden_layer_activation="relu",
        output_activation="sigmoid",
        loss_func=keras.losses.BinaryCrossentropy(),
        opt_func=keras.optimizers.Adam(),
        metrics=["AUC"],
        validation_split=0.3,
        random_seed=69,
    ):
        """
        docstring
        """
        self.X_test = X_test
        self.y_test = y_test
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=random_seed
        )
        self.modelwrapper = ModelWrapper(
            inputs=keras.Input(shape=(X_train.shape[1],)),
            outputs=keras.layers.Dense(1, activation=output_activation),
        )
        self.layers = layers
        self.loss_func = loss_func
        self.opt_func = opt_func
        self.metrics = metrics
        self.output_activation = output_activation
        # with open("test.txt", "a+") as fh:
        #     print(loss_func, file=fh)
        self.model_name = f"{len(layers)}x{hidden_layer_activation}x{output_activation}layer opt-{opt_func._name} loss-{loss_func.name}"
        self.loss_func_name = loss_func.name
        self.opt_func_name = opt_func._name
        for layer in layers:
            # print(layer)
            self.modelwrapper.add_layer(layer)
        self.modelwrapper.build_model()
        self.modelwrapper.compile_model(loss_func, opt_func, metrics)
        print(self.model_name)
        print(self.opt_func)

    def fit_predict_model(self, validation_data=None, **kwargs):
        if validation_data == "validation":
            vali = (self.X_val, self.y_val)
        self.modelwrapper.fit(
            self.X_train, self.y_train, validation_data=vali, **kwargs
        )
        self.y_preds = self.modelwrapper.predict(self.X_test)

    def summarise_model_instance(self, path="data/reporting/nnmodels/"):
        dt = datetime.now().strftime("%Y%m%d%H%M%S")
        # print(dt)
        path = path + f"{dt}-{self.model_name}/"
        if not os.path.exists(path):
            os.makedirs(path)
        self.modelwrapper.plot_model(path + "model_plot.png")
        if self.output_activation == "sigmoid":
            fpr, tpr, _ = roc_curve(self.y_test, self.y_preds)
            self.fpr = fpr
            self.tpr = tpr
            # print(fpr[5])
            # print(tpr[5])
            plt.plot([0, 1], [0, 1], linestyle="--", label="")
            plt.plot(
                fpr,
                tpr,
                linestyle="-",
                label=self.output_activation + " Activation Output",
                alpha=0.8,
            )
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.savefig(path + "roc_curve.png")
            plt.close()
            self.y_predslabels = np.multiply(self.y_preds > 0.5, 1)
            self.modelwrapper.plot_epoch_loss("AUC", "AUC", path + "epoch_AUC.png")
            self.modelwrapper.plot_epoch_loss(path=path + "epoch_loss.png")
            self.roc_auc_score = roc_auc_score(self.y_test, self.y_preds)
            self.f1_score = f1_score(self.y_test, self.y_predslabels)
            with open(path + "report.txt", "a+") as f:
                f.write(self.modelwrapper.__repr__())
                f.write(f"\nroc_auc_score: {self.roc_auc_score}\n\n")
                try:
                    f.write(classification_report(self.y_test, self.y_predslabels))
                except ValueError:
                    f.write("\nCould Not Produce Classification Report")
            self.r2 = None
            self.mse = None
        elif self.output_activation is None:
            for i in range(self.X_test.shape[1]):
                plt.scatter(self.X_test[:, i], self.y_test, marker=".", label="Actual")
                plt.scatter(
                    self.X_test[:, i], self.y_preds, marker=".", label="Prediction"
                )
                plt.xlabel(f"Feature{i+1}")
                plt.ylabel("Response")
                plt.legend()
                plt.savefig(path + f"Feat{i+1}_ac_pred.png")
                plt.clf()
            with open(path + "report.txt", "a+") as f:
                f.write(self.modelwrapper.__repr__())
                self.r2 = r2_score(self.y_test, self.y_preds)
                f.write(f"R2: {self.r2}\n")
                self.mse = mean_squared_error(self.y_test, self.y_preds)
                f.write(f"MSE: {self.mse}\n")
            self.fpr = None
            self.tpr = None
            self.roc_auc_score = None
            self.f1_score = None


def summarise_model_instances(
    models: List[ModelInstance], hidden_layer_sizes, layers_length
):
    fprs = []
    tprs = []
    roc_auc_scores = []
    f1_scores = []
    opt_func = []
    loss_func = []
    r2 = []
    mse = []
    for model in models:
        fprs.append(model.fpr)
        tprs.append(model.tpr)
        roc_auc_scores.append(model.roc_auc_score)
        f1_scores.append(model.f1_score)
        opt_func.append(model.opt_func_name)
        loss_func.append(model.loss_func_name)
        mse.append(model.mse)
        r2.append(model.r2)
    return pd.DataFrame(
        {
            "fprs": str(fprs),
            "tprs": str(tprs),
            "roc_auc_scores": roc_auc_scores,
            "f1_scores": f1_scores,
            "r2": r2,
            "mse": mse,
            "opt_func": opt_func,
            "loss_func": loss_func,
            "hidden_layer_size": hidden_layer_sizes,
            "layers_length": layers_length,
        }
    )
