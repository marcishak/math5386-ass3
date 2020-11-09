import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz, export_text
# from sklearn
from datetime import datetime
import os
from typing import List
from tensorflow import keras


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
        sklearn_model,
        feature_names,
        rstate,
    ):
        """
        docstring
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = sklearn_model
        self.model_name = sklearn_model.__repr__().replace("()","")
        self.feature_names = feature_names


    def fit_predict_model(self):
        self.model.fit(self.X_train,self.y_train)
        self.y_preds = self.model(self.X_test)
        self.model_struct = export_text(self.model, feature_names=self.feature_names)

    def build_classifcation_report(self, path="data/reporting/sklmodels/"):
        dt = datetime.now().strftime("%Y%m%d%H%M%S")
        # print(dt)
        path = path + f"{dt}-{self.model_name}/"
        if not os.path.exists(path):
            os.makedirs(path)
        # self.modelwrapper.plot_model(path + "model_plot.png")
        fpr, tpr, _ = roc_curve(self.y_test, self.y_preds)
        self.fpr = fpr
        self.tpr = tpr
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
        # self.y_predslabels = np.multiply(self.y_preds > 0.5, 1)
        self.roc_auc_score = roc_auc_score(self.y_test, self.y_preds)
        self.f1_score = f1_score(self.y_test, self.y_preds)
        with open(path + "report.txt", "a+") as f:
            f.write(self.model_struct)
            f.write(f"\nroc_auc_score: {self.roc_auc_score}\n\n")
            try:
                f.write(
                    classification_report(
                        self.y_test, self.y_pred
                    )
                )
            except ValueError:
                f.write("\nCould Not Produce Classification Report")

    def plot_model(self, path):




def summarise_model_instances(models: List[ModelInstance], hidden_layer_size, layers_length):
    fprs = []
    tprs = []
    roc_auc_scores = []
    f1_scores = []
    opt_func = []
    loss_func = []
    for model in models:
        fprs.append(model.fpr)
        tprs.append(model.tpr)
        roc_auc_scores.append(model.roc_auc_score)
        f1_scores.append(model.f1_score)
        opt_func.append(model.opt_func_name)
        loss_func.append(model.loss_func_name)
    return {
        "fprs": str(fprs),
        "tprs": str(tprs),
        "roc_auc_scores": roc_auc_scores,
        "f1_scores": f1_scores,
        "opt_func": opt_func, 
        "loss_func": loss_func,
        "hidden_layer_size": [hidden_layer_size] * len(models),
        "layers_length": [layers_length] * len(models),
    }
