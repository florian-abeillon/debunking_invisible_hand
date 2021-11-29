""" src/display """

import pandas as pd
import seaborn as sns


def plot_avg_q_table(agents: list) -> None:
    """ Displays average learnt Q-table across sellers """
    df_concat = pd.concat([ agent.get_q_table() for agent in agents ])
    df_mean = df_concat.groupby(df_concat.index).mean()
    sns.heatmap(
        df_mean, 
        cmap='jet_r', 
        cbar=True
    )

# Display sellers' results

def plot_prices(sellers: list) -> None:
    """ Displays price fluctuations across buyers """
    df_prices = pd.DataFrame({
        seller.name: [ price for _, price, _ in seller.get_history() ]
        for seller in sellers
    })
    df_prices.plot()
    
# # Display buyers' results

# def plot_prices(buyers: list) -> None:
#     """ Displays average learnt Q-table across buyers """
#     df_prices = pd.DataFrame({
#         seller.name: [ price for price, _ in seller.get_history() ]
#         for seller in sellers
#     })
#     df_prices.plot()
    