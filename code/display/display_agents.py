""" display/display_agents """

import pandas as pd
import seaborn as sns
from src.utils import plot_q_table, update_epsilon


def plot_avg_q_table(agents: list) -> None:
    """ Displays average learnt Q-table across sellers """
    df_concat = pd.concat([ agent.get_q_table() for agent in agents ])
    df_avg = df_concat.groupby(df_concat.index).mean()
    plot_q_table(df_avg)


def plot_epsilon(epsilon: float, size_unk: int, nb_rounds: int) -> None:
    """ Displays evolution of curiosity (epsilon factor) """

    y_lim = 10 * nb_rounds
    epsilon_values = []
    proportion = 1
    for _ in range(y_lim):
        epsilon_values.append(update_epsilon(proportion, epsilon))
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
        ylabel="Curiosity factor (epsilon)",
        xlim=(0, y_lim),
        ylim=(0, 1)
    )
    fig.lines[1].set_linestyle("--")
    fig.lines[2].set_linestyle("--")
