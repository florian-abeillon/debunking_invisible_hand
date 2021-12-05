""" main """

from tqdm import tqdm

from display import (plot_avg_budget, plot_avg_nb_purchases, plot_avg_q_table,
                     plot_avg_sub_q_tables, plot_avg_variations)
from src.constants import NB_ROUNDS, NB_TO_MATCH, VERBOSE
from src.simulate import get_new_agents, play_round

if __name__ == "__main__":
    sellers, buyers = get_new_agents(Verbose=VERBOSE)

    iterator = tqdm(range(NB_ROUNDS)) if VERBOSE else range(NB_ROUNDS)
    for _ in iterator:
        sellers, buyers = play_round(sellers, buyers, nb_to_match=NB_TO_MATCH, Verbose=VERBOSE)

    # Sellers
    plot_avg_variations(sellers)
    plot_avg_q_table(sellers)
    
    # Buyers
    plot_avg_budget(buyers)
    plot_avg_nb_purchases(buyers)
    plot_avg_sub_q_tables(buyers)
