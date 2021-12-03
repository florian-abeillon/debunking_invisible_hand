""" display/display_sellers """

from typing import List, Tuple, Union

import numpy as np
import seaborn as sns
from agents.constants import PRICE_PROD
from matplotlib import pyplot as plt

from display.display_agents import plot_avg


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
        sum_or_not_sum = lambda sales: sales if avg else sum(sales)
        func = lambda qty_prod, price, sales: price * sum_or_not_sum(sales) - qty_prod * price_prod
        y_label = "Profit"

    if avg:
        y_label = f"Average {y_label.lower()}"

    n = len(history)
    y = [ 
        func(qty_prod, price, sales) 
        for qty_prod, price, sales in history
    ]
    fig = sns.lineplot(
        x=range(n), 
        y=y
    )
    n_mean = n // 100
    fig_mean = sns.lineplot(
        x=[ (i + 0.5) * n_mean for i in range(100) ],
        y=[
            np.mean(y[i * n_mean:(i + 1) * n_mean])
            for i in range(100)
        ]
    )

    fig.set(
        xlabel="Rounds",
        ylabel=y_label
    )
    plt.show()
    

def plot_avg_variations(sellers: list, value: Union[str, List[str]] = [ 'price', 'qty', 'profit' ]) -> None:
    """ Displays price/qty/profit fluctuations across buyers """
    extract_from_hist = lambda hist_round, _: ( hist_round[0], hist_round[1], sum(hist_round[2]) )
    plot_value = lambda history: plot_variations(history, value=value, avg=True)
    plot_avg(sellers, plot_fct=plot_value, extract_from_hist=extract_from_hist)
    