""" agents/Seller/utils """

import pandas as pd


def get_q_table(price_min: int, price_max: int, qty_min: int, qty_max: int) -> pd.DataFrame:
    """ Create a buyer's Q-table, initialized with zeros """
    data = {
        qty: pd.arrays.SparseArray([ 0 for _ in range(price_min, price_max+1) ]) 
        for qty in range(qty_min, qty_max+1)
    }

    q_table = pd.DataFrame(data)
    q_table.index = range(price_min, price_max+1)
    q_table.columns = range(qty_min, qty_max+1)
    return q_table
    

def get_q_table_size(price_min: int, price_max: int, qty_min: int, qty_max: int) -> int:
    """ Return a seller's Q-table number of cells """
    return (price_max - price_min) * (qty_max - qty_min)
