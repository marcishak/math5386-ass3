import argparse

import numpy as np

import data_eng as de
import data_sci as ds


def run_de_pipe(dataset: str):
    if dataset == "park":
        df = de.data_clean.run_dc_pipe_park("data/raw/park/train_data.txt")
        X_train, X_test, y_train, y_test = de.feature_eng.run_fe_pipe_park(df)
        return X_train, X_test, y_train, y_test
    elif dataset == "abalone":
        df = de.data_clean.run_dc_pipe_abelone("data/raw/abalone/abalone.data")
        X_train, X_test, y_train, y_test = de.feature_eng.run_fe_pipe_abalone(df)
        return X_train, X_test, y_train, y_test
    else:
        raise ValueError("Dataset must be park or abalone")


def run_ds_pipe(X_train=None, X_test=None, y_train=None, y_test=None, dataset=None):
    if dataset is not None:
        X_train = np.loadtxt("data/featured/" + dataset + "X_train.csv")
        y_train = np.loadtxt("data/featured/" + dataset + "y_train.csv")
        X_test = np.loadtxt("data/featured/" + dataset + "X_test.csv")
        y_test = np.loadtxt("data/featured/" + dataset + "y_test.csv")
    ds.run_pipe(X_train, X_test, y_train, y_test)


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
    for dataset in datasets:
        if args.pipeline == "process":
            run_de_pipe(dataset)
        elif args.pipeline == "model":
            run_ds_pipe(dataset=dataset)
        else:
            run_ds_pipe(run_de_pipe(dataset))


if __name__ == "__main__":
    main()
