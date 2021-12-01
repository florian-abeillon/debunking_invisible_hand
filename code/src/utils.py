""" agents.utils """

import math

import matplotlib.pyplot as plt
import seaborn as sns


def update_epsilon(proportion: float, epsilon: float, steepness: int = 10, break_point: float = 0.5) -> float:
    """ Computes next value of curiosity (epsilon), depending on proportion frac """
    return epsilon + (1 - epsilon) / (1 + math.exp(- steepness * (proportion - break_point)))


def plot_q_table(df, title: str = "") -> None:
    """ Displays heatmap of Q-table """
    fig = sns.heatmap(
        df.sort_index(ascending=False), 
        cmap='jet_r', 
        cbar=True
    )
    fig.set(
        xlabel="Quantity",
        ylabel="Price"
    )
    if title:
        fig.set(title=title)
    plt.show()
