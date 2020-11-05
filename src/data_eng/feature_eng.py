import pandas as pd
import numpy as np
from typing import Any
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

def _split_features_response(df:pd.DataFrame, y_col:Any=28, x_cols:Any=None):
    y = df.loc[:,y_col]
    if x_cols is not None:
        df.iloc[:,x_cols]
    else:
        X = df.drop(columns=[y_col])
    return (X,y)


def _sklearn_feature_pipeline(X:np.array):
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('PCA', PCA(n_components=6)),
        ])
    return pipeline.fit_transform(X)

def _write_return_files(X_train:np.array, X_test:np.array, y_train:np.array, y_test:np.array, path:str = "data/featured/"):
    np.savetxt(path + "X_train.csv", X_train, delimiter=",")
    np.savetxt(path + "y_train.csv", y_train, delimiter=",")
    np.savetxt(path + "X_test.csv", X_test, delimiter=",")
    np.savetxt(path + "y_test.csv", y_test, delimiter=",")
    return X_train, X_test, y_train, y_test


def run_fe_pipe(df:pd.DataFrame, test_prop:float = 0.2, random_seed:float = 69):
    X,y = _split_features_response(df)
    X = _sklearn_feature_pipeline(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_prop, random_state=random_seed
    )
    return _write_return_files(X_train, X_test, y_train, y_test)





