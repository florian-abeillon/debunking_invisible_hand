""" display/display_buyers """

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from agents.constants import BUDGET, PRICE_MAX, PRICE_MIN
from agents.utils import get_avg_q_table
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from display.display_agents import plot_avg, running_avg


def plot_variations(history: List[int], 
                    value: Union[str, List[str]] = [ 'budget', 'nb_purchases' ],
                    budget: int = BUDGET,
                    avg: bool = False) -> None:
    """ 
        Display budget/nb_purchases fluctuations 
    """

    if type(value) == list:
        for v in value:
            plot_variations(history, value=v, budget=budget, avg=avg)
        return

    assert value in [ 'budget', 'nb_purchases' ], f"value={value} should be within [ 'budget', 'nb_purchases' ]"
    
    if value == 'budget':
        func = lambda hist_round: hist_round[-1][0] if hist_round else budget 
        y_label = "Budget leftovers"
    else:
        func = lambda hist_round: sum([ qty for _, _, qty in hist_round ])
        y_label = "Number of purchases"

    x_lim = len(history)
    y = history if avg else [ func(hist_round) for hist_round in history ]
    # Plot fluctuations
    fig = sns.lineplot(
        x=range(x_lim), 
        y=y
    )

    x_avg, y_avg = running_avg(y)
    # Plot mean of fluctuations (with a 100th window)
    fig_mean = sns.lineplot(
        x=x_avg,
        y=y_avg
    )

    legend_labels = [ "Actual", "Running average" ]

    if value == 'budget':
        fig_baseline = sns.lineplot(
            x=[ 0, len(history) ], 
            y=[ budget, budget ]
        )
        fig_baseline.lines[2].set_linestyle("--")
        legend_labels.append("Initial value")

    y_min, y_max = np.min(history), np.max(history)
    y_lim = (
        0 if y_min > 0 else 1.1 * y_min,
        1.1 * y_max if y_max > 0 else 0
    )
    fig.set(
        ylim=y_lim,
        title=f"{y_label} over rounds",
        xlabel="Rounds",
        ylabel=y_label
    )
    plt.legend(labels=legend_labels)
    plt.show()


def plot_avg_budget(buyers: list, 
                    non_zero: bool = True) -> None:
    """ 
        Display average budget fluctuations over buyers 
    """
    extract_from_hist = lambda hist_round, buyer: hist_round[-1][0] if hist_round else buyer.budget 
    plot_value = lambda history: plot_variations(history, value='budget', avg=True)
    plot_avg(buyers, plot_fct=plot_value, extract_from_hist=extract_from_hist, non_zero=non_zero)


def plot_avg_nb_purchases(buyers: list) -> None:
    """ 
        Display average budget fluctuations over buyers 
    """
    extract_from_hist = lambda hist_round, _: sum([ transac[2] for transac in hist_round ])
    plot_value = lambda history: plot_variations(history, value='nb_purchases', avg=True)
    plot_avg(buyers, plot_fct=plot_value, extract_from_hist=extract_from_hist, non_zero=True)



def plot_w_slider(frames = List[go.Frames], 
                  x_label: str = "", 
                  y_label: str = "") -> None:
    """ 
        Plot Plotly figures with a slider 
    """

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


def plot_history(history: List[List[Tuple[int, int, int]]],
                 budget: int) -> None:
    """ 
        Display purchases history (quantity purchased over price), for each budget 
    """
    
    # Create dict with purchases for every possible budget
    d = { i: [] for i in range(budget + 1) }
    for i, hist_round in enumerate(history):
        for budget, price, qty in hist_round:
            d[budget].append(( i, price, qty ))
    d = { budget: np.array(transac) for budget, transac in d.items() }
    
    frames = [
        go.Frame(
            data=go.Scatter(
                x=transac[:, 2], 
                y=transac[:, 1],
                mode='markers',
                marker_color=transac[:, 0]
            ) if np.any(transac) else go.Scatter(x=[], y=[]),
            name=str(budget)
        )
        for budget, transac in d.items()
    ]

    frames.reverse()
    plot_w_slider(frames, x_label="Quantity purchased", y_label="Price offered")


def plot_sub_q_tables(q_table: np.array) -> None:
    """ 
        Display interactive heatmap of learnt Q-table, for each budget 
    """

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

    frames.reverse()
    plot_w_slider(frames, x_label="Quantity purchased", y_label="Price offered")
    

def plot_avg_sub_q_tables(buyers: list) -> None:
    """ 
        Display heatmap of average learnt Q-tables, for each budget 
    """
    plot_sub_q_tables(get_avg_q_table(buyers))
    


def plot_demand_curve(q_table: np.array) -> tuple:
    """ 
        Display demand curve from learnt Q-table, for each budget 
    """

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
    ax.set(
        title="Demand curve",
        xlabel="Quantity",
        ylabel="Price"
    )
    
    p, = ax.plot(
        demand[BUDGET], 
        prices, 
        'y',
        label="Actual"
    )
    lim, = ax.plot(
        demand_lim[BUDGET],
        prices, 
        'y1',
        label="Limit"
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
    """ 
        Display average demand curve from learnt Q-tables, for each budget 
    """
    return plot_demand_curve(get_avg_q_table(buyers))
