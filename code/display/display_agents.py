""" display/display_agents """

from typing import Callable, List

import numpy as np
import seaborn as sns
from agents import Agent
from agents.utils import get_avg_q_table
from matplotlib import pyplot as plt
from src.utils import update_epsilon


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



def plot_epsilon(epsilon: float, size_unk: int, nb_rounds: int) -> None:
    """ Display evolution of curiosity (epsilon factor) """

    x_lim = 10 * nb_rounds if nb_rounds < 0.25 * size_unk else max(2.5 * size_unk, 1.1 * nb_rounds)
    x_lim = int(x_lim)
    epsilon_values, proportion_values = [], []
    proportion = 1.

    for _ in range(x_lim):
        epsilon_values.append(update_epsilon(proportion, epsilon))
        proportion_values.append(proportion)
        proportion -= epsilon_values[-1] / size_unk

    fig = sns.lineplot(
        x=range(x_lim), 
        y=epsilon_values
    )
    fig_baseline = sns.lineplot(
        x=[ 0, x_lim ], 
        y=[ epsilon, epsilon ]
    )
    fig_game = sns.lineplot(
        x=[ nb_rounds, nb_rounds ], 
        y=[ 0, epsilon_values[nb_rounds] ]
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
