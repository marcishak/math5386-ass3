from typing import Any, List

import numpy as np
import pandas as pd


def _drop_cols(df: pd.DataFrame, list_cols: List[Any]):
    return df.drop(columns=list_cols)


def _write_cleaned_data(
    df: pd.DataFrame, path: str = "data/cleaned/park/clean_data.csv"
):
    df.to_csv(path, index=False)
    return df


def run_dc_pipe_park(path: str):
    df = pd.read_csv(path, header=None)
    df_ret = df.pipe(_drop_cols, [0, 22, 27]).pipe(_write_cleaned_data)
    return df_ret

def run_dc_pipe_abelone(path: str):
    """
    stub because no cleaning is needed
    """
    df = pd.read_csv(path, header=None)
    df_ret = df.pipe(_write_cleaned_data, "data/cleaned/abalone/clean_data.csv")
    return df_ret

