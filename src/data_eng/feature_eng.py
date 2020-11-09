from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def _split_features_response(df: pd.DataFrame, y_col: Any = 28, x_cols: Any = None):
    y = df.loc[:, y_col]
    if x_cols is not None:
        df.iloc[:, x_cols]
    else:
        X = df.drop(columns=[y_col])
    return (X, y)


# I could merge these but I don't want to in case I want to add to one dataset pipeline and not the other!
def _sklearn_feature_pipeline_park(X: np.array) -> np.array:
    pipeline = Pipeline([("scaler", MinMaxScaler()), ("PCA", PCA(n_components=6)),])
    return pipeline.fit_transform(X)


def _sklearn_feature_pipeline_abalone(X: np.array) -> np.array:
    pipeline = Pipeline([("scaler", MinMaxScaler()), ("PCA", PCA(n_components=1)),])
    return pipeline.fit_transform(X)


def _to_df_get_dummies(X: np.array, prefix: str) -> pd.DataFrame:
    return pd.get_dummies(X, prefix=prefix)


def _rbind_1_np_array_df(df: pd.DataFrame, X: np.array, X_name: str) -> pd.DataFrame:
    # print(df)
    # print(X)
    s = pd.Series(X[:, 0], name=X_name)
    return pd.concat([df, s], axis=1)


def _write_return_files(
    X_train: np.array,
    X_test: np.array,
    y_train: np.array,
    y_test: np.array,
    path: str = "data/featured/",
):
    np.savetxt(path + "X_train.csv", X_train, delimiter=",")
    np.savetxt(path + "y_train.csv", y_train, delimiter=",")
    np.savetxt(path + "X_test.csv", X_test, delimiter=",")
    np.savetxt(path + "y_test.csv", y_test, delimiter=",")
    return X_train, X_test, y_train, y_test


def run_fe_pipe_park(df: pd.DataFrame, test_prop: float = 0.2, random_seed: float = 69):
    X, y = _split_features_response(df)
    X = _sklearn_feature_pipeline_park(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_prop, random_state=random_seed
    )
    return _write_return_files(X_train, X_test, y_train, y_test, "data/featured/park/")


def run_fe_pipe_abalone(
    df: pd.DataFrame, test_prop: float = 0.2, random_seed: float = 69
):
    X, y = _split_features_response(df, y_col=8)
    X, sex = _split_features_response(df, y_col=0)
    X = _sklearn_feature_pipeline_abalone(X)
    sex = _to_df_get_dummies(sex, "Sex")
    X = _rbind_1_np_array_df(sex, X, "pca").values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_prop, random_state=random_seed
    )
    return _write_return_files(
        X_train, X_test, y_train, y_test, "data/featured/abalone/"
    )

