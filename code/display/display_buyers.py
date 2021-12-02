""" display/display_buyers """

from typing import List, Union

import numpy as np
from agents import Buyer
from src.utils import plot_q_table


def plot_avg_sub_q_tables(buyers: List[Buyer], budgets: Union[int, List[int]] = []) -> None:
    """ Displays heatmap of average learnt Q-table, for each budget """
    q_table_avg = np.mean([ buyer.get_q_table() for buyer in buyers ], axis=0)

    if not budgets:
        budgets = range(q_table_avg.shape[0])
    elif type(budgets) == int:
        budgets = [budgets]

    for budget in budgets:
        sub_q_table_avg = q_table_avg[budget]
        title = f"Sub-Q-table for budget={budget}"
        plot_q_table(sub_q_table_avg, title=title)


# TODO
# def plot_prices(buyers: list) -> None:
#     """ Displays average learnt Q-table across buyers """
#     df_prices = pd.DataFrame({
#         seller.name: [ price for price, _ in seller.get_history() ]
#         for seller in sellers
#     })
#     df_prices.plot()
