""" agents.utils """

import math

from src.constants import BREAKPOINT_EPSILON, STEEPNESS_EPSILON


def update_epsilon(proportion: float, epsilon: float, steepness: int = STEEPNESS_EPSILON, break_point: float = BREAKPOINT_EPSILON) -> float:
    """ Computes next value of curiosity (epsilon), depending on proportion frac """
    return epsilon + (1 - epsilon) / (1 + math.exp(- steepness * (proportion - break_point)))
