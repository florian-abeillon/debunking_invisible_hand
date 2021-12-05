""" agents/Buyer/utils """

import numpy as np


def get_q_table(budget_max: int, 
                price_min: int, 
                price_max: int, 
                qty_max: int) -> np.array:
    """ 
        Create a buyer's Q-table, initialized with zeros 
    """
    assert price_min > 0, f"price_min={price_min} should be positive (no free item)"
    nb_budget, nb_price, nb_qty = budget_max + 1, price_max - price_min + 1, qty_max + 1
    
    # Initialize Q-table with NaNs
    a = np.full(( nb_budget, nb_price, nb_qty ), np.nan)
    # Fill *actually* used cells with 0.
    for budget in range(nb_budget):
        for price in range(price_min, price_max + 1):
            qty_lim = min(qty_max, budget // price) + 1
            idx_price = price - price_min
            a[budget, idx_price, :qty_lim] = 0.

    return a
    

def get_q_table_size(budget_max: int, price_min: int, price_max: int, qty_max: int):
    """ 
        Return a buyer's Q-table number of cells *actually* used (as all cells with budget < qty * price are not used)
    """
    counter = 0
    for budget in range(budget_max + 1):
        for price in range(price_min, price_max + 1):
            counter += min(qty_max, budget // price) + 1
    return counter
