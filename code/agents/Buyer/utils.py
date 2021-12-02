""" agents/Buyer/utils """

import numpy as np


def get_q_table(budget_max: int, price_min: int, price_max: int, qty_max: int) -> np.array:
    """ Create a buyer's Q-table, initialized with zeros """
    # TODO: Try with sparse arrays?
    assert price_min > 0, f"price_min={price_min} should be positive (no free item)"
    a = np.full((budget_max + 1, price_max - price_min + 1, qty_max + 1), np.nan)
    for budget in range(budget_max + 1):
        for price in range(price_min, price_max + 1):
            idx_price = price - price_min
            qty_lim = min(qty_max, budget // price) + 1
            a[budget, idx_price, :qty_lim] = 0.
    return a
    

def get_q_table_size(budget_max: int, price_min: int, price_max: int, qty_max: int):
    """ 
        Return a buyer's Q-table number of cells *actually* used 
        -> All cells with budget < qty * price are not used
    """
    return len([
        None
        for budget in range(0, budget_max + 1)
        for price in range(price_min, price_max + 1)
        for qty in range(0, qty_max + 1)
        if qty * price <= budget
    ])
