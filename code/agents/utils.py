""" agents/utils """

import numpy as np


def get_avg_q_table(agents: list) -> np.array:
    """ 
        Return average Q-table from list of Agents 
    """
    return np.mean([ agent.get_q_table() for agent in agents ], axis=0)
