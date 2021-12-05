""" display/display_sellers """

from typing import List, Tuple, Union

import seaborn as sns
from agents.constants import PRICE_MAX, PRICE_MIN
from matplotlib import pyplot as plt
from src.constants import SAVE_PREFIX

from display.display_agents import plot_avg, plot_variations

UTILS = {
    'qty': (
        lambda hist_round, seller: hist_round[0],
        "Selling price"
    ),
    'price': (
        lambda hist_round, seller: hist_round[1],
        "Produced quantity"
    ),
    'sales': (
        lambda hist_round, seller: sum(hist_round[2]),
        "Sold quantity"
    ),
    'profit': (
        lambda hist_round, seller: hist_round[1] * sum(hist_round[2]) - hist_round[0] * seller.price_prod,
        "Profit"
    )
}


def plot_history(history: List[Tuple[int, int, List[int]]],
                 price_prod: int = 0,
                 save: bool = False,
                 save_prefix: str = SAVE_PREFIX,
                 save_suffix: str = "") -> None:
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

    if save:
        save_name = "seller_history"
        if save_prefix:
            save_name = f"{save_prefix}_{save_name}"
        if save_suffix:
            save_name = f"{save_name}_{save_suffix}"
        plt.savefig(f"results/{save_name}.png")


def plot_variations_sellers(history: List[Tuple[int, int, List[int]]], 
                            value: str,
                            save: bool = False,
                            save_prefix: str = SAVE_PREFIX,
                            save_suffix: str = "") -> None:
    """ 
        Display sellers fluctuations 
    """
    assert value in [ 'qty', 'price', 'sales', 'profit' ], f"value={value} should be within [ 'qty', 'price', 'sales', 'profit' ]"
    extract_fct, y_label = UTILS[value]
    history = [ 
        extract_fct(hist_round, None)
        for hist_round in history 
    ]
    plot_variations(history, y_label, save=save, save_prefix=save_prefix, save_suffix=save_suffix)


def plot_avg_variations_sellers(sellers: list,
                                value: Union[str, List[str]] = [ 'qty', 'price', 'sales', 'profit' ],
                                save: bool = False,
                                save_prefix: str = SAVE_PREFIX,
                                save_suffix: str = "") -> None:
    """ 
        Display average sellers fluctuations
    """
    if type(value) == list:
        for v in value:
            plot_avg_variations_sellers(sellers, value=v, save=save)
        return
    assert value in [ 'qty', 'price', 'sales', 'profit' ], f"value={value} should be within [ 'qty', 'price', 'sales', 'profit' ]"
    plot_avg(sellers, *UTILS[value], save=save, save_prefix=save_prefix, save_suffix=save_suffix)
    