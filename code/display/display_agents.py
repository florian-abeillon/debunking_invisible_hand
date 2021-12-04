""" display/display_agents """

from typing import Callable, List

import numpy as np
import seaborn as sns
from agents import Agent
from agents.Buyer.utils import get_q_table_size as get_size_unk_buyer
from agents.constants import (BANDIT_BREAKPOINT_BUYER,
                              BANDIT_BREAKPOINT_SELLER, BANDIT_STEEPNESS_BUYER,
                              BANDIT_STEEPNESS_SELLER, BUDGET_MAX,
                              CURIOSITY_BUYER, CURIOSITY_SELLER, PRICE_MAX,
                              PRICE_MIN, QTY_MAX, QTY_MIN)
from agents.Seller.utils import get_q_table_size as get_size_unk_seller
from agents.utils import get_avg_q_table
from matplotlib import pyplot as plt
from src.constants import NB_SELLERS
from src.utils import update_curiosity


def plot_avg(agents: List[Agent], plot_fct: Callable, extract_from_hist: Callable, **kwargs) -> None:
    """ Display average budget fluctuations over buyers """
    history_concat = np.array([
        [ 
            extract_from_hist(hist_round, agent)
            for hist_round in agent.get_history(**kwargs)
        ]
        for agent in agents
    ])
    history_mean = history_concat.mean(axis=0)
    plot_fct(history_mean)


def plot_q_table(a: np.array, title: str = "") -> None:
    """ Display heatmap of learnt Q-table """
    fig = sns.heatmap(
        a, 
        cmap='jet_r', 
        cbar=True
    )
    kwargs = { 'title': title } if title else {}
    fig.set(
        xlabel="Quantity",
        ylabel="Price",
        **kwargs
    )
    fig.invert_yaxis()
    plt.show()


def plot_avg_q_table(agents: List[Agent]) -> None:
    """ Display heatmap of average learnt Q-table across sellers """
    plot_q_table(get_avg_q_table(agents))



def plot_epsilon(type_agent: str, nb_rounds: int) -> None:
    """ Display evolution of curiosity (epsilon factor) """
    assert type_agent in [ 'buyer', 'seller' ], f"type_agent={type_agent} should be within [ 'buyer', 'seller' ]"

    if type_agent == "seller":
        size_unk = get_size_unk_seller(PRICE_MIN, PRICE_MAX, QTY_MIN, QTY_MAX)
        epsilon = CURIOSITY_SELLER
        steepness = BANDIT_STEEPNESS_SELLER
        breakpoint = BANDIT_BREAKPOINT_SELLER
    else:
        size_unk = get_size_unk_buyer(BUDGET_MAX, PRICE_MIN, PRICE_MAX, QTY_MAX)
        epsilon = CURIOSITY_BUYER
        steepness = BANDIT_STEEPNESS_BUYER
        breakpoint = BANDIT_BREAKPOINT_BUYER
        nb_rounds =  int(nb_rounds * NB_SELLERS / 2)

    x_lim = 10 * nb_rounds if nb_rounds < 0.25 * size_unk else int(max(2.5 * size_unk, 1.1 * nb_rounds))
    curiosity_values, proportion_values = [], []
    proportion = 1.

    for _ in range(x_lim):
        curiosity = update_curiosity(proportion, epsilon, steepness, breakpoint)
        curiosity_values.append(curiosity)
        proportion_values.append(proportion)
        proportion -= curiosity / size_unk

    fig = sns.lineplot(
        x=range(x_lim), 
        y=curiosity_values
    )
    fig_baseline = sns.lineplot(
        x=[ 0, x_lim ], 
        y=[ epsilon, epsilon ]
    )
    fig_game = sns.lineplot(
        x=[ nb_rounds, nb_rounds ], 
        y=[ 0, curiosity_values[nb_rounds] ]
    )

    fig.set(
        xlabel="Rounds",
        ylabel="Estimation of epsilon (curiosity factor)",
        xlim=(0, x_lim),
        ylim=(0, 1)
    )
    fig.lines[1].set_linestyle("--")
    fig.lines[2].set_linestyle("--")
    plt.show()
