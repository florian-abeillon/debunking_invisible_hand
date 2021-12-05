""" display/display_agents """

from typing import Callable, List, Tuple

import numpy as np
import seaborn as sns
from agents import Agent
from agents.constants import CURIOSITY_BUYER, CURIOSITY_SELLER
from agents.utils import get_avg_q_table
from matplotlib import pyplot as plt


def plot_avg(agents: List[Agent], 
             plot_fct: Callable, 
             extract_from_hist: Callable, 
             **kwargs) -> None:
    """ 
        Display average budget fluctuations over buyers 
    """
    # Extract relevant info from Agents histories
    history_concat = np.array([
        [ 
            extract_from_hist(hist_round, agent)
            for hist_round in agent.get_history(**kwargs)
        ]
        for agent in agents
    ])
    # Compute mean
    history_mean = history_concat.mean(axis=0)
    # Plot mean
    plot_fct(history_mean)


def plot_q_table(a: np.array, title: str = "") -> None:
    """ 
        Display heatmap of learnt Q-table 
    """
    fig = sns.heatmap(
        a, 
        cmap='jet_r', 
        cbar=True
    )
    kwargs = { 'title': title } if title else {}
    fig.set(
        title="Learnt Q-table",
        xlabel="Quantity",
        ylabel="Price",
        **kwargs
    )
    fig.invert_yaxis()
    plt.show()


def plot_avg_q_table(agents: List[Agent]) -> None:
    """ 
        Display heatmap of average learnt Q-table across sellers 
    """
    plot_q_table(get_avg_q_table(agents))



def plot_curiosity(curiosity_values: List[float], 
                   epsilon: float = None) -> None:
    """ 
        Display evolution of curiosity 
    """
    x_lim = len(curiosity_values)
    
    # Display curiosity evolution
    fig = sns.lineplot(
        x=range(x_lim),
        y=curiosity_values
    )
    if epsilon is not None:
        # Display curiosity asymptot (epsilon)
        fig_baseline = sns.lineplot(
            x=[ 0, x_lim ], 
            y=[ epsilon, epsilon ]
        )
        fig.lines[1].set_linestyle("--")

    fig.set(
        title="Evolution of curiosity over rounds",
        xlabel="Rounds",
        ylabel="Curiosity",
        xlim=( 0, x_lim ),
        ylim=( 0, 1 )
    )
    plt.legend(labels=[ "Curiosity", "Epsilon (asymptot)" ])
    plt.show()


def plot_avg_curiosity(agents: List[Agent]) -> None:
    """ 
        Display evolution of curiosity on average
    """
    type_agent = agents[0].get_type()
    assert type_agent in [ 'buyer', 'seller' ], f"type_agent={type_agent} should be within [ 'buyer', 'seller' ]"
    
    if type_agent == 'buyer':
        epsilon = CURIOSITY_BUYER
        n_min = min([
            len(buyer.get_curiosity())
            for buyer in agents
        ])
        func = lambda agent: agent.get_curiosity()[:n_min]
    else:
        epsilon = CURIOSITY_SELLER
        func = lambda agent: agent.get_curiosity()

    # Concatenate agents' curiosity histories
    curiosity_concat = np.array([
        func(agent) for agent in agents
    ])
    # Compute mean
    curiosity_mean = curiosity_concat.mean(axis=0)
    # Plot mean
    plot_curiosity(curiosity_mean, epsilon=epsilon)


def running_avg(y: list, 
                step: int = 100) -> list:
    x_avg, y_avg = [], []
    step = int(len(y) / step)
    idx_start, idx_end = -step // 2, step // 2

    for i in range(0, len(y), step):
        window = y[max(0, idx_start):idx_end]
        avg = sum(window) / len(window)
        x_avg.append(i)
        y_avg.append(avg)

        idx_start += step
        idx_end += step

    return x_avg, y_avg
