""" src/display/display_buyers """

from typing import List, Union

import pandas as pd
from src.utils import plot_q_table


def plot_avg_sub_q_tables(buyers: list, budgets: Union[int, List[int]] = []) -> None:
    """ Displays heatmap of average learnt Q-table, for each budget """
    df_concat = pd.concat([ buyer.get_q_table() for buyer in buyers ])
    df_avg = df_concat.groupby(df_concat.index).mean()
    df_avg.index = pd.MultiIndex.from_tuples(df_avg.index)

    if budgets:
        if type(budgets) == int:
            budgets = [budgets]
    else:
        budgets = df_avg.index.levels[0]

    for budget in budgets:
        sub_df_avg = df_avg.loc[budget].copy(deep=True)
        sub_df_avg.loc[:, 1:100] = sub_df_avg.loc[:, 1:100][sub_df_avg.loc[:, 1:100]!=0.]
        sub_df_avg.dropna(axis=1, how='all')

        title = f"Sub-Q-table for budget={budget}"
        plot_q_table(sub_df_avg, title=title)


# TODO
# def plot_prices(buyers: list) -> None:
#     """ Displays average learnt Q-table across buyers """
#     df_prices = pd.DataFrame({
#         seller.name: [ price for price, _ in seller.get_history() ]
#         for seller in sellers
#     })
#     df_prices.plot()
