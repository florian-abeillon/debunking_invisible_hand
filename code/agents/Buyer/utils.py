""" agents/Buyer/utils """

import pandas as pd


def get_q_table(budget_max: int, price_min: int, price_max: int, qty_max: int) -> pd.DataFrame:
    """ Create a buyer's Q-table, initialized with zeros """
    # Indexes are pairs (budget_left, price)
    index_list = [ 
        ( i, j ) 
        for i in range(0, budget_max+1) 
        for j in range(price_min, price_max+1) 
    ]
    return pd.DataFrame(
        index=pd.MultiIndex.from_tuples(index_list), 
        columns=range(0, qty_max+1), 
        dtype=float
    ).fillna(0.)
    

def get_q_table_size(budget_max: int, price_min: int, price_max: int, qty_max: int):
    """ 
        Return a buyer's Q-table number of cells *actually* used 
        -> All cells with budget < qty * price are not used
    """
    counter = 0
    for budget in range(0, budget_max+1):
        for price in range(price_min, price_max+1):
            for qty in range(0, qty_max+1):
                if qty * price <= budget:
                    counter += 1
    return counter
