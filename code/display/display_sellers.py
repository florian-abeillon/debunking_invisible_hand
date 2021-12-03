""" display/display_sellers """

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from agents.Seller.constants import PRICE_PROD


def plot_variations(history: List[Tuple[int, int, List[int]]], 
                    value: Union[str, List[str]] = [ 'price', 'qty', 'profit' ], 
                    price_prod: int = PRICE_PROD,
                    avg: bool = False) -> None:
    """ Displays price/qty/profit fluctuations """

    if type(value) == list:
        for v in value:
            plot_variations(history, value=v, price_prod=price_prod, avg=avg)
        return

    assert value in [ 'price', 'qty', 'profit' ], f"value={value} should be within [ 'price', 'qty', 'profit' ]"
    
    if value == 'price':
        func = lambda qty_prod, price, sales: price
        y_label = "Selling price"
    elif value == 'qty':
        func = lambda qty_prod, price, sales: qty_prod
        y_label = "Produced quantity"
    else:
        if avg:
            func = lambda qty_prod, price, sales: price * sales - qty_prod * price_prod
        else:
            func = lambda qty_prod, price, sales: price * sum(sales) - qty_prod * price_prod
        y_label = "Profit"

    if avg:
        y_label = f"Average {y_label.lower()}"

    fig = sns.lineplot(
        x=range(len(history)), 
        y=[ 
            func(qty_prod, price, sales) 
            for qty_prod, price, sales in history
        ]
    )
    fig.set(
        xlabel="Rounds",
        ylabel=y_label
    )
    plt.show()
    

def plot_avg_variations(sellers: list, value: Union[str, List[str]] = [ 'price', 'qty', 'profit' ]) -> None:
    """ Displays price/qty/profit fluctuations across buyers """
    history_concat = np.array([ 
        [
            ( qty_prod, price, sum(sales) )
            for qty_prod, price, sales in seller.get_history()
        ]        
        for seller in sellers 
    ])
    history_mean = history_concat.mean(axis=0)
    plot_variations(history_mean, value=value, avg=True)
    