""" agents.utils """

import math


def update_curiosity(proportion: float, 
                     epsilon: float, 
                     steepness: int, 
                     break_point: float) -> float:
    """ 
        Compute next value of curiosity (epsilon), depending on proportion frac 
    """
    curiosity = 1 / (1 + math.exp(- steepness * (proportion - break_point)))
    return epsilon + (1 - epsilon) * curiosity
