""" display/display_sellers """

from typing import List, Tuple, Union

import numpy as np
import seaborn as sns
from agents.constants import PRICE_MAX, PRICE_MIN, PRICE_PROD
from matplotlib import pyplot as plt

from display.display_agents import plot_avg


def plot_history(history: List[Tuple[int, int, List[int]]],
                 price_prod: int = 0) -> None:
    """ 
        Display sales history 
    """
    
    prices, nb_sales = [], []
    for _, price, sales in history:
        prices.append(price)
        nb_sales.append(sum(sales))

    x_lim = len(history)
    y_lim = 100 / PRICE_MIN
    alpha = 1 - 0.8 * min(1, x_lim / 50000)

    # Display nb_sales over selling prices
    fig = sns.scatterplot(
        x=prices,
        y=nb_sales,
        hue=list(range(x_lim)),
        palette='jet_r',
        alpha=alpha
    )
    if price_prod:
        # Display production price limit
        fig_baseline = sns.lineplot(
            x=[ price_prod, price_prod ], 
            y=[ 0, y_lim ]
        )

    fig.set(
        title="Number of sales over chosen selling price",
        xlim=( PRICE_MIN, PRICE_MAX ),
        ylim=( 0, y_lim ),
        xlabel="Selling price",
        ylabel="Number of sales"
    )
    plt.show()


def plot_variations(history: List[Tuple[int, int, List[int]]], 
                    value: Union[str, List[str]] = [ 'price', 'qty', 'profit' ], 
                    price_prod: int = PRICE_PROD,
                    avg: bool = False) -> None:
    """ 
        Display price/qty/profit fluctuations 
    """

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

    x_lim = len(history)
    y = [ 
        func(qty_prod, price, sales) 
        for qty_prod, price, sales in history
    ]
    # Plot fluctuations
    fig = sns.lineplot(
        x=range(x_lim), 
        y=y
    )
    
    x_step = x_lim // 100
    # Plot mean of fluctuations (with a 100th window)
    fig_mean = sns.lineplot(
        x=[ (i + 0.5) * x_step for i in range(100) ],
        y=[
            np.mean(y[i * x_step:(i + 1) * x_step])
            for i in range(100)
        ]
    )

    y_min, y_max = np.min(y), np.max(y)
    y_lim = [
        0 if y_min > 0 else 1.1 * y_min,
        1.1 * y_max if y_max > 0 else 0
    ]
    fig.set(
        title=f"{y_label} over rounds",
        ylim=y_lim,
        xlabel="Rounds",
        ylabel=y_label
    )
    plt.legend(labels=[ "Actual", "Average" ])
    plt.show()
    

def plot_avg_variations(sellers: list, 
                        value: Union[str, List[str]] = [ 'price', 'qty', 'profit' ]) -> None:
    """ 
        Display price/qty/profit fluctuations across buyers 
    """
    extract_from_hist = lambda hist_round, _: ( hist_round[0], hist_round[1], sum(hist_round[2]) )
    plot_value = lambda history: plot_variations(history, value=value, avg=True)
    plot_avg(sellers, plot_fct=plot_value, extract_from_hist=extract_from_hist)
    