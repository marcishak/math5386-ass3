import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow import keras
import sklearn.tree as tree

class ModelWrapper:
    """
    Wrapper for defining and testing SkLearn Models
    """
    def __init__(self):
        pass


def fit_plot_roc(mod, X_test, y_test, path=None):
    """
    TODO: docstring
    """
    y_preds = mod.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_preds)
    plt.plot([0, 1], [0, 1], linestyle="--", label="")
    # calculate roc curve for model
    # plot model roc curve
    plt.plot(fpr, tpr, linestyle="-", label="Sigmoid Activation Output", alpha=0.8)
    # axis labels
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # show the legend
    plt.legend()
    # show the plot
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    return f"roc_auc_score: {roc_auc_score(y_test, y_preds)}"
