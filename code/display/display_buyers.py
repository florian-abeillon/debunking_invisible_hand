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
    demand_lim = [
        [ 
            budget // price
            for price in range(PRICE_MIN, PRICE_MAX + 1) 
        ]
        for budget in range(BUDGET + 1)
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
    lim, = ax.plot(
        demand_lim[BUDGET],
        prices, 
        'y1'
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
        lim.set_xdata(demand_lim[budget])
        fig.canvas.draw()
    
    plt.show()
    return slider, update_slider


def plot_avg_demand_curve(buyers: list) -> tuple:
    """ Display average demand curve from learnt Q-tables, for each budget """
    return plot_demand_curve(get_avg_q_table(buyers))



def plot_variations(history_variations: List[int], y_label: str = "") -> None:
    """ Display budget fluctuations """
    x_lim = len(history_variations)
    x_step = x_lim // 100

    fig = sns.lineplot(
        x=range(x_lim), 
        y=history_variations
    )
    fig_mean = sns.lineplot(
        x=[ (i + 0.5) * x_step for i in range(100) ],
        y=[
            np.mean(history_variations[i * x_step:(i + 1) * x_step])
            for i in range(100)
        ]
    )

    y_min, y_max = np.min(history_variations), np.max(history_variations)
    y_lim = [
        0 if y_min > 0 else 1.1 * y_min,
        1.1 * y_max if y_max > 0 else 0
    ]
    fig.set(
        ylim=y_lim,
        xlabel="Rounds",
        ylabel=y_label
    )


def plot_budget(history_budget: List[int], budget: int = BUDGET) -> None:
    """ Display budget fluctuations """
    plot_variations(history_budget, y_label="Budget left at the end of a round")
    fig_baseline = sns.lineplot(
        x=[ 0, len(history_budget) ], 
        y=[ budget, budget ]
    )
    fig_baseline.lines[2].set_linestyle("--")
    plt.show()

def plot_avg_budget(buyers: list, non_zero: bool = True) -> None:
    """ Display average budget fluctuations over buyers """
    extract_from_hist = lambda hist_round, buyer: hist_round[-1][0] if hist_round else buyer.budget 
    plot_avg(buyers, plot_fct=plot_budget, extract_from_hist=extract_from_hist, non_zero=non_zero)



def plot_nb_purchases(history_nb_purchases: List[int]) -> None:
    """ Display budget fluctuations """
    plot_variations(history_nb_purchases, y_label="Number of purchases")
    plt.show()

def plot_avg_nb_purchases(buyers: list) -> None:
    """ Display average budget fluctuations over buyers """
    extract_from_hist = lambda hist_round, _: sum([ transac[2] for transac in hist_round ])
    plot_avg(buyers, plot_fct=plot_nb_purchases, extract_from_hist=extract_from_hist, non_zero=True)
