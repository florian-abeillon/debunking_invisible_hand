""" display/display_buyers """

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from agents.Buyer.constants import BUDGET
from agents.constants import PRICE_MAX, PRICE_MIN, QTY_MAX
from agents.utils import get_avg_q_table


def plot_w_slider(frames = List[go.Frames], x_label: str = "", y_label: str = ""):
    """ Plot Plotly figures with a slider """

    kwargs = {
        'sliders': [{ 'steps': [
            {
                'args': [
                    [frame.name],
                    {
                        'frame': {
                            'duration': 0, 
                            'redraw': True
                        },
                        'mode': "immediate"
                    }
                ],
                'label': frame.name, 
                'method': "animate"
            }
            for frame in frames
        ]}]
    }
    if x_label:
        kwargs['xaxis'] = { 'title': x_label }
    if y_label:
        kwargs['yaxis'] = { 'title': y_label }

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(**kwargs)
    fig.show()



def plot_sub_q_tables(q_table: np.array) -> None:
    """ Display interactive heatmap of learnt Q-table, for each budget """

    zmin, zmax = np.nanmin(q_table), np.nanmax(q_table)
    frames = []

    for i, sub_q_table in enumerate(q_table):
        df = pd.DataFrame(sub_q_table)
        heatmap = go.Heatmap(
            x=df.columns, 
            y=df.index,
            z=df.values,
            zmin=zmin,
            zmax=zmax
        )
        frames.append(go.Frame(data=heatmap, name=i))

    plot_w_slider(frames, x_label="Quantity purchased", y_label="Price offered")
    

def plot_avg_sub_q_tables(buyers: list) -> None:
    """ Display heatmap of average learnt Q-tables, for each budget """
    plot_sub_q_tables(get_avg_q_table(buyers))
    


def plot_demand_curve(q_table: np.array) -> None:
    """ Display demand curve from learnt Q-table, for each budget """
    prices = list(range(PRICE_MIN, PRICE_MAX + 1))
    frames = [
        go.Frame(
            data=go.Scatter(
                x=np.argmax(sub_q_table, axis=1), 
                y=prices
            ), 
            layout=go.Layout(yaxis={ 'range': [ 0, QTY_MAX ] }),        # TODO: Put everything on the same scale
            name=i
        )
        for i, sub_q_table in enumerate(q_table)
    ]
    plot_w_slider(frames, x_label="Quantity", y_label="Price")


def plot_avg_demand_curve(buyers: list) -> None:
    """ Display average demand curve from learnt Q-tables, for each budget """
    plot_demand_curve(get_avg_q_table(buyers))



def plot_budget(history_budget: List[int], budget: int = BUDGET) -> None:
    """ Display budget fluctuations """
    x_lim = len(history_budget)

    fig = sns.lineplot(
        x=range(x_lim), 
        y=history_budget
    )
    fig_baseline = sns.lineplot(
        x=[ 0, x_lim ], 
        y=[ budget, budget ]
    )

    fig.set(
        xlabel="Rounds",
        ylabel="Budget left at the end of a round"
    )
    fig.lines[1].set_linestyle("--")
    plt.show()


def plot_avg_budget(buyers: list, non_zero: bool = True) -> None:
    """ Display average budget fluctuations over buyers """
    history_budget_concat = np.array([
        [ 
            hist_round[-1][0] if hist_round else buyer.budget 
            for hist_round in buyer.get_history(non_zero=non_zero)
        ]
        for buyer in buyers
    ])
    history_budget_mean = history_budget_concat.mean(axis=0)
    plot_budget(history_budget_mean)



def plot_nb_purchases(history_nb_purchases: List[int]) -> None:
    """ Display budget fluctuations """
    fig = sns.lineplot(
        x=range(len(history_nb_purchases)), 
        y=history_nb_purchases
    )
    fig.set(
        xlabel="Rounds",
        ylabel="Number of purchases"
    )
    plt.show()


def plot_avg_nb_purchases(buyers: list) -> None:
    """ Display average budget fluctuations over buyers """
    history_nb_purchases_concat = np.array([
        [ 
            sum([ transac[2] for transac in hist_round ])
            for hist_round in buyer.get_history(non_zero=True)
        ]
        for buyer in buyers
    ])
    history_nb_purchases_mean = history_nb_purchases_concat.mean(axis=0)
    plot_nb_purchases(history_nb_purchases_mean)
