""" agents/utils """

from typing import Callable, Tuple, Union

import pandas as pd


def update_sparse(df: pd.DataFrame, 
                  index: Union[int, Tuple[int, int]], 
                  column: int, 
                  update: Callable) -> None:
    """ Convert row (pd.arrays.SparseArray) to dense, change value and converts back to sparse structure """
    dtype_col = df.dtypes[column]
    df[column] = df[column].sparse.to_dense()
    df.loc[index, column] = update(df.loc[index, column])
    df[column] = df[column].astype(dtype_col)
