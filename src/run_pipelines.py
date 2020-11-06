import data_eng as de
import data_sci as ds
import argparse


def run_de_pipe(path = "data/raw/train_data.txt"):
    df = de.data_clean.run_dc_pipe("data/raw/train_data.txt")
    X_train, X_test, y_train, y_test = de.feature_eng.run_fe_pipe(df)
    return X_train, X_test, y_train, y_test

def run_ds_pipe():
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', choices=['process', 'model', 'all'], default = "all",
                        help='which pipeline would you like to run?')
    args = parser.parse_args()
    if args.pipeline == "process":
        run_de_pipe()
    elif args.pipeline == "model":
        run_ds_pipe()
    else:
        run_ds_pipe(run_de_pipe())

if __name__ == "__main__":
    main()