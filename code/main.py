""" main """

from tqdm import tqdm

from display import (plot_avg_curiosity, plot_avg_q_table,
                     plot_avg_sub_q_tables, plot_avg_variations_buyers,
                     plot_avg_variations_sellers)
from src.constants import NB_ROUNDS, NB_TO_MATCH, VERBOSE
from src.simulate import get_new_agents, play_round

if __name__ == "__main__":
    sellers, buyers = get_new_agents(Verbose=VERBOSE)

    for _ in tqdm(range(NB_ROUNDS)):
        sellers, buyers = play_round(sellers, buyers, nb_to_match=NB_TO_MATCH, Verbose=VERBOSE)

    # Sellers
    sellers[0].plot_history()
    plot_avg_variations_sellers(sellers)
    plot_avg_q_table(sellers)
    plot_avg_curiosity(sellers)
    
    # Buyers
    buyers[0].plot_history()
    plot_avg_variations_buyers(buyers)
    plot_avg_sub_q_tables(buyers)
    plot_avg_curiosity(buyers)
