""" display/display_sellers """

from typing import List, Tuple, Union

import numpy as np
import seaborn as sns
from agents.constants import PRICE_MAX, PRICE_PROD
from matplotlib import pyplot as plt
plt.style.use('ggplot')


from display.display_agents import plot_avg

def moving_average(a, n=1000) -> np.ndarray:
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_variations(history: List[Tuple[int, int, List[int]]], 
                    value: Union[str, List[str]] = [ 'price', 'qty', 'profit', 'sales' ], 
                    price_prod: int = PRICE_PROD,
                    avg: bool = False) -> None:
    """ Displays price/qty/profit fluctuations """

    if type(value) == list:
        for v in value:
            plot_variations(history, value=v, price_prod=price_prod, avg=avg)
        return

    assert value in [ 'price', 'qty', 'profit', 'sales' ], f"value={value} should be within [ 'price', 'qty', 'profit', 'sales' ]"
    
    if value == 'price':
        func = lambda qty_prod, price, sales: price
        y_label = "Selling price"
    elif value == 'qty':
        func = lambda qty_prod, price, sales: qty_prod
        y_label = "Produced quantity"
    elif value == 'sales':
        func = lambda qty_prod, price, sales: sales
        y_label = "Sold quantity"
    else:
        sum_or_not_sum = lambda sales: sales if avg else sum(sales)
        func = lambda qty_prod, price, sales: price * sum_or_not_sum(sales) - qty_prod * price_prod
        y_label = "Profit"

    if avg:
        y_label = f"Average {y_label.lower()}"

    x_lim = len(history)
    y = [ 
        func(qty_prod, price, sales) 
        for qty_prod, price, sales in history
    ]
    fig = sns.lineplot(
        x=range(x_lim), 
        y=y
    )
    
    x_step = x_lim // 100
    fig_mean = sns.lineplot(
        # x=[ (i + 0.5) * x_step for i in range(100) ],
        # y=[
        #     np.mean(y[i * x_step:(i + 1) * x_step])
        #     for i in range(100)
        # ]
        x=range(moving_average(y).shape[0]),
        y=moving_average(y)
    )

    y_min, y_max = np.min(y), np.max(y)
    y_lim = [
        0 if y_min > 0 else 1.1 * y_min,
        1.1 * y_max if y_max > 0 else 0
    ]
    fig.set(
        ylim=y_lim,
        xlabel="Rounds",
        ylabel=y_label
    )
    plt.show()
    

def plot_avg_variations(sellers: list, value: Union[str, List[str]] = [ 'price', 'qty', 'profit', 'sales' ]) -> None:
    """ Displays price/qty/profit fluctuations across buyers """
    extract_from_hist = lambda hist_round, _: ( hist_round[0], hist_round[1], sum(hist_round[2]) )
    plot_value = lambda history: plot_variations(history, value=value, avg=True)
    plot_avg(sellers, plot_fct=plot_value, extract_from_hist=extract_from_hist)
    