""" agents/Seller/utils """

import numpy as np


def get_q_table(price_min: int, price_max: int, qty_min: int, qty_max: int) -> np.array:
    """ Create a seller's Q-table, initialized with zeros """
    return np.zeros((price_max - price_min + 1, qty_max - qty_min + 1))
    

def get_q_table_size(price_min: int, price_max: int, qty_min: int, qty_max: int) -> int:
    """ Return a seller's Q-table number of cells """
    return (price_max - price_min) * (qty_max - qty_min)
