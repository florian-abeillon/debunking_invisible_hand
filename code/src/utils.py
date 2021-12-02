""" agents.utils """

import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def update_epsilon(proportion: float, epsilon: float, steepness: int = 10, break_point: float = 0.5) -> float:
    """ Computes next value of curiosity (epsilon), depending on proportion frac """
    return epsilon + (1 - epsilon) / (1 + math.exp(- steepness * (proportion - break_point)))


def plot_q_table(a: np.array, title: str = "") -> None:
    """ Displays heatmap of Q-table """
    fig = sns.heatmap(
        a, 
        cmap='jet_r', 
        cbar=True
    )
    fig.set(
        xlabel="Quantity",
        ylabel="Price"
    )
    fig.invert_yaxis()
    if title:
        fig.set(title=title)
    plt.show()
