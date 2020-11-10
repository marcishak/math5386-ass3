import argparse

import numpy as np

import data_eng as de
import data_sci as ds
from typing import Optional, List

def run_de_pipe(dataset: str):
    if dataset == "park":
        df = de.data_clean.run_dc_pipe_park("data/raw/park/train_data.txt")
        X_train, X_test, y_train, y_test = de.feature_eng.run_fe_pipe_park(df)
        return [X_train, y_train, X_test, y_test]
    elif dataset == "abalone":
        df = de.data_clean.run_dc_pipe_abelone("data/raw/abalone/abalone.data")
        X_train, X_test, y_train, y_test = de.feature_eng.run_fe_pipe_abalone(df)
        return [X_train, y_train, X_test, y_test]
    else:
        raise ValueError("Dataset must be park or abalone")


def run_ds_pipe(train_test_list:Optional[List[np.array]], p_type:str, dataset:str=None):
    if dataset is not None:
        X_train = np.loadtxt("data/featured/" + dataset + "X_train.csv")
        y_train = np.loadtxt("data/featured/" + dataset + "y_train.csv")
        X_test = np.loadtxt("data/featured/" + dataset + "X_test.csv")
        y_test = np.loadtxt("data/featured/" + dataset + "y_test.csv")
    else:
        X_train, y_train, X_test, y_test = train_test_list
    ds.run_pipe(X_train, y_train, X_test, y_test, p_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline",
        choices=["process", "model", "all"],
        default="all",
        help="which pipeline would you like to run?",
    )
    parser.add_argument(
        "--dataset",
        choices=["abalone", "park", "all"],
        default="all",
        help="which dataset would you like to run it on?",
    )
    args = parser.parse_args()
    if args.dataset == "all":
        datasets = ["abalone", "park"]
    else:
        datasets = [args.dataset]
    models = []
    if "abalone" in datasets:
        models.append("DecisionTreeRegressor")
        models.append("RandomForestRegressor")
    if "park" in datasets:
        models.append("DecisionTreeClassifier")
        models.append("RandomForestClassifier")
    m_ds = list(zip(sorted(datasets*2), models))
    # m_ds = list(zip(datasets, models))
    print(m_ds)
    for dataset, model in m_ds:
        # print(dataset)
        # print(model)
        if args.pipeline == "process":
            run_de_pipe(dataset)
        elif args.pipeline == "model":
            run_ds_pipe(None, dataset=dataset, p_type = model)
        else:
            run_ds_pipe(run_de_pipe(dataset),p_type = model)


if __name__ == "__main__":
    main()
