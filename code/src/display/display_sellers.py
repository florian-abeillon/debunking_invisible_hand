""" src/display/display_sellers """

import pandas as pd


def plot_prices(sellers: list, avg: bool = False) -> None:
    """ Displays price fluctuations across buyers """
    
    df_prices = pd.DataFrame({
        seller.name: [ price for _, price, _ in seller.get_history() ]
        for seller in sellers
    })
    y_label = "Selling price"

    if avg:
        df_prices = df_prices.mean(axis=1)
        y_label = "Average selling price"

    df_prices.plot(
        xlabel="Rounds",
        ylabel=y_label
    )
    