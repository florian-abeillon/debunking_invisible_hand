""" display/display_agents """

from typing import List

import numpy as np
import seaborn as sns
from agents import Agent
from src.utils import plot_q_table, update_epsilon


def plot_avg_q_table(agents: List[Agent]) -> None:
    """ Displays average learnt Q-table across sellers """
    q_table_avg = np.mean([ agent.get_q_table() for agent in agents ], axis=0)
    plot_q_table(q_table_avg)


def plot_epsilon(epsilon: float, size_unk: int, nb_rounds: int) -> None:
    """ Displays evolution of curiosity (epsilon factor) """

    y_lim = 10 * nb_rounds if nb_rounds < 0.25 * size_unk else max(2.5 * size_unk, 1.1 * nb_rounds)
    y_lim = int(y_lim)
    epsilon_values, proportion_values = [], []
    proportion = 1.

    for _ in range(y_lim):
        epsilon_values.append(update_epsilon(proportion, epsilon))
        proportion_values.append(proportion)
        proportion -= epsilon_values[-1] / size_unk

    fig = sns.lineplot(
        x=range(y_lim), 
        y=epsilon_values
    )
    fig_baseline = sns.lineplot(
        x=[ 0, y_lim ], 
        y=[ epsilon, epsilon ]
    )
    fig_game = sns.lineplot(
        x=[ nb_rounds, nb_rounds ], 
        y=[ 0, epsilon_values[nb_rounds] ]
    )

    fig.set(
        xlabel="Rounds",
        ylabel="Estimation of epsilon (curiosity factor)",
        xlim=(0, y_lim),
        ylim=(0, 1)
    )
    fig.lines[1].set_linestyle("--")
    fig.lines[2].set_linestyle("--")
