""" main """

from tqdm import tqdm

from src.simulate import get_new_agents, play_round

if __name__ == "__main__":
    sellers, buyers = get_new_agents()
    for _ in tqdm(range(1000)):
        sellers, buyers = play_round(sellers, buyers)
