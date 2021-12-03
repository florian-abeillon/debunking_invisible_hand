""" display/display_buyers """

from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from agents.constants import BUDGET, PRICE_MAX, PRICE_MIN
from agents.utils import get_avg_q_table
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from display.display_agents import plot_avg


def plot_w_slider(frames = List[go.Frames], x_label: str = "", y_label: str = "") -> None:
    """ Plot Plotly figures with a slider """

    layout = {
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
        layout['xaxis'] = { 'title': x_label }
    if y_label:
        layout['yaxis'] = { 'title': y_label }

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(**layout)
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
    


def plot_demand_curve(q_table: np.array) -> tuple:
    """ Display demand curve from learnt Q-table, for each budget """

    demand = [
        np.argmax(sub_q_table, axis=1)
        for sub_q_table in q_table
    ]
    prices = list(range(PRICE_MIN, PRICE_MAX + 1))
    
    fig = plt.figure()
    plt.subplots_adjust(bottom=0.25)
    ax = fig.subplots()
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Price")
    p, = ax.plot(
        demand[BUDGET], 
        prices, 
        'y'
    )
    
    ax_slide = plt.axes([ 0.25, 0.1, 0.65, 0.03 ])
    slider = Slider(
        ax_slide, 
        'Budget',
        0, 
        BUDGET, 
        valinit=BUDGET, 
        valstep=1
    )

    def update_slider(_):
        budget = slider.val
        p.set_xdata(demand[budget])
        fig.canvas.draw()
    
    plt.show()
    return slider, update_slider


def plot_avg_demand_curve(buyers: list) -> tuple:
    """ Display average demand curve from learnt Q-tables, for each budget """
    return plot_demand_curve(get_avg_q_table(buyers))



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
    extract_from_hist = lambda hist_round, buyer: hist_round[-1][0] if hist_round else buyer.budget 
    plot_avg(buyers, plot_fct=plot_budget, extract_from_hist=extract_from_hist, non_zero=non_zero)



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
    extract_from_hist = lambda hist_round, _: sum([ transac[2] for transac in hist_round ])
    plot_avg(buyers, plot_fct=plot_nb_purchases, extract_from_hist=extract_from_hist, non_zero=True)
