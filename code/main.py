""" main """

from tqdm import tqdm

from src.constants import NB_ROUNDS
from src.simulate import get_new_agents, play_round

if __name__ == "__main__":
    Verbose = True
    sellers, buyers = get_new_agents(Verbose=Verbose)

    iterator = tqdm(range(NB_ROUNDS)) if Verbose else range(NB_ROUNDS)
    for _ in iterator:
        sellers, buyers = play_round(sellers, buyers, Verbose=Verbose)
