""" agents/utils """

from typing import List

import numpy as np

from agents.Agent import Agent


def get_avg_q_table(agents: List[Agent]) -> np.array:
    """ Return average Q-table from list of Agents """
    return np.mean([ agent.get_q_table() for agent in agents ], axis=0)
