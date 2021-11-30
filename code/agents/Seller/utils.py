""" agents/Seller/utils """

import pandas as pd


def get_q_table(price_min: int, price_max: int, qty_min: int, qty_max: int) -> pd.DataFrame:
    """ Create a buyer's Q-table, initialized with zeros """
    return pd.DataFrame(
        index=range(price_min, price_max+1), 
        columns=range(qty_min, qty_max+1), 
        dtype=float
    ).fillna(0.)
    

def get_q_table_size(price_min: int, price_max: int, qty_min: int, qty_max: int) -> int:
    """ Return a seller's Q-table number of cells """
    return (price_max - price_min) * (qty_max - qty_min)
