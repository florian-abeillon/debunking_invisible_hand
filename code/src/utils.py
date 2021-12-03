""" agents.utils """

import math


def update_epsilon(proportion: float, epsilon: float, steepness: int = 10, break_point: float = 0.5) -> float:
    """ Computes next value of curiosity (epsilon), depending on proportion frac """
    return epsilon + (1 - epsilon) / (1 + math.exp(- steepness * (proportion - break_point)))
