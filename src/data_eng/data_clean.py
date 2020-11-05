import pandas as pd
import numpy as np
from typing import List, Any


def _drop_cols(df: pd.DataFrame, list_cols: List[Any]):
    return df.drop(columns=list_cols)


def _write_cleaned_data(df: pd.DataFrame, path: str = "data/cleaned/data_clean.csv"):
    df.to_csv(path, index=False)
    return df


def run_dc_pipe(path: str):
    df = pd.read_csv(path, header=None)
    df_ret = (
        df
        .pipe(_drop_cols, [0, 22, 27])
        .pipe(_write_cleaned_data)
        )
    return df_ret

