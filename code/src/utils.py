""" agents.utils """

import matplotlib.pyplot as plt
import seaborn as sns


def update_epsilon(epsilon: float, proportion: float) -> float:
    """ Computes next value of curiosity (epsilon), depending on proportion frac """
    # TODO: Find better function
    return (1 - epsilon) * proportion + epsilon


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
