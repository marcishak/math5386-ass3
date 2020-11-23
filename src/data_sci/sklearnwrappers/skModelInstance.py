import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_curve,
    roc_auc_score,
    f1_score,
    r2_score,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeRegressor,
    plot_tree,
    DecisionTreeClassifier,
    export_text,
    export_graphviz,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from mlxtend.plotting import plot_decision_regions
import pandas as pd
from datetime import datetime
import os
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
        sklearn_model,
        feature_names,
        optimisation_level=100,
        val_prop=0.3,
    ):
        """
        docstring
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_preds = None
        self.X_val = None
        self.y_val = None
        self.model = sklearn_model
        self.model_name = sklearn_model.__repr__().replace("()", "")
        self.feature_names = feature_names
        self.rstate = np.random.default_rng(seed=69)
        self.optimisation_level = optimisation_level
        self.val_prop = val_prop
        self.ccp_path = None
        self.model_struct = None
        self.fpr = None
        self.tpr = None
        self.roc_auc_score = None
        self.f1_score = None
        self.r2 = None
        self.mse = None
        self.sample_ccp_alphas = None
        self.fitted_models = None
        self.train_scores = None 
        self.test_scores  = None


    def fit_predict_model(self):
        if "Random" not in self.model_name:
            self._fit_optimised_model()
        self.model.fit(self.X_train, self.y_train)
        self.y_preds = self.model.predict(self.X_test)
        if "Random" not in self.model_name:
            self.model_struct = export_text(self.model, feature_names=self.feature_names)
        else:
            self.model_struct = "Random Forest Model\n"

    def summarise_model_instance(self, path="data/reporting/sklmodels/"):
        dt = datetime.now().strftime("%Y%m%d%H%M%S")
        # print(dt)
        path = path + f"{dt}-{self.model_name}/"
        if not os.path.exists(path):
            os.makedirs(path)
        # self.modelwrapper.plot_model(path + "model_plot.png")
        if "Classifier" in self.model_name:
            # print(/)
            # fpr, tpr, _ = roc_curve(self.y_test, self.y_preds)
            fpr, tpr, _ = roc_curve(self.y_test, self.model.predict_proba(self.X_test)[:,1])
            self.fpr = fpr
            self.tpr = tpr
            plt.plot([0, 1], [0, 1], linestyle="--", label="")
            plt.plot(
                fpr, tpr, linestyle="-", label=self.model_name, alpha=0.8,
            )
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.savefig(path + "roc_curve.png")
            plt.close()
            # self.roc_auc_score = roc_auc_score(self.y_test, self.y_preds)
            self.roc_auc_score = roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:,1])
            self.f1_score = f1_score(self.y_test, self.y_preds)
            with open(path + "report.txt", "a+") as f:
                f.write(self.model_struct)
                f.write(f"\nroc_auc_score: {self.roc_auc_score}\n\n")
                try:
                    f.write(classification_report(self.y_test, self.y_preds))
                except ValueError:
                    f.write("\nCould Not Produce Classification Report")
            plot_decision_regions(
                self.X_test,
                self.y_test.values,
                self.model,
                filler_feature_values={x: 0 for x in range(2, self.X_test.shape[1])},
                filler_feature_ranges={
                    x: max(self.X_test.reshape(-1, 1))[0]
                    for x in range(2, self.X_test.shape[1])
                },
            )
            plt.savefig(path + "decision_regions.png")
            plt.clf()
        elif "Regressor" in self.model_name:
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
                f.write(self.model_struct)
                self.r2 = r2_score(self.y_test, self.y_preds)
                self.mse = mean_squared_error(self.y_test, self.y_preds)
                f.write(f"R2: {self.r2}\n")
                f.write(f"MSE: {self.mse}\n")
        if "Random" not in self.model_name:
            self._plot_model(path)

    def _plot_model(self, path):
        """
        As seen in: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
        """
        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs alpha for training and validation sets")
        ax.plot(
            self.sample_ccp_alphas,
            self.train_scores,
            marker="o",
            label="train",
            drawstyle="steps-post",
        )
        ax.plot(
            self.sample_ccp_alphas,
            self.test_scores,
            marker="o",
            label="validate",
            drawstyle="steps-post",
        )
        ax.legend()
        # plt.show()
        plt.savefig(path + "alpha_plot.png")
        with open(path + "graphviz.dot", "a+") as f:
            try:
                f.write(export_graphviz(self.model, filled=True, leaves_parallel=True))
            except:
                print("GRAPHVIZ export failed")
        plt.clf()

    def _fit_optimised_model(self):
        """
        As seen in: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
        """
        # print(self.X_train.shape)
        # print(self.y_train.shape)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train,
            self.y_train,
            test_size=self.val_prop,
            random_state=self.rstate.integers(1000),
        )
        self.model.fit(self.X_train, self.y_train)
        self.ccp_path = self.model.cost_complexity_pruning_path(
            self.X_train, self.y_train
        )
        self.sample_ccp_alphas = np.linspace(
            min(self.ccp_path.ccp_alphas),
            max(self.ccp_path.ccp_alphas),
            num=self.optimisation_level,
        )
        self.fitted_models = []
        for ccp_alpha in self.sample_ccp_alphas:
            clf = eval(
                f"{self.model_name}(random_state={self.rstate.integers(1000)}, ccp_alpha={ccp_alpha})"
            )
            clf.fit(self.X_train, self.y_train)
            self.fitted_models.append(clf)
        self.train_scores = [
            clf.score(self.X_train, self.y_train) for clf in self.fitted_models
        ]
        self.test_scores = [
            clf.score(self.X_val, self.y_val) for clf in self.fitted_models
        ]
        self.model = self.fitted_models[
            self.test_scores.index(
                [x for x in self.test_scores if x == max(self.test_scores)][0]
            )
        ]


def summarise_model_instances(models: List[ModelInstance], model_library="sklearn"):
    if "Classifier" in models[0].model_name:
        fprs = []
        tprs = []
        roc_auc_scores = []
        f1_scores = []
        model_type = []
        for model in models:
            # print(model)
            fprs.append(model.fpr)
            tprs.append(model.tpr)
            roc_auc_scores.append(model.roc_auc_score)
            f1_scores.append(model.f1_score)
            model_type.append(model.model_name)
        return pd.DataFrame(
            {
                "fprs": str(fprs),
                "tprs": str(tprs),
                "roc_auc_scores": roc_auc_scores,
                "f1_scores": f1_scores,
                "model_type": model_type,
                "model_library": model_library,
            }
        )
    elif "Regressor" in models[0].model_name:
        model_type = []
        mse = []
        r2 = []
        for model in models:
            # print(model)
            mse.append(model.mse)
            model_type.append(model.model_name)
            r2.append(model.r2)
        return pd.DataFrame(
            {
                "mse": mse,
                "model_type": model_type,
                "r2": r2,
                "model_library": model_library,
            }
        )

