from typing import Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_valid_test(df: pd.DataFrame, stratify_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # specify the columns to use for splitting the data

    # split the data into training and temp sets
    train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df[stratify_cols])

    # split the temp set into validation and test sets
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[stratify_cols])

    return train_df, val_df, test_df
