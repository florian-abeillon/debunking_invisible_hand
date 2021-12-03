""" agents/Agent """

from typing import Union
from uuid import uuid4

import numpy as np
from src.utils import update_epsilon


class Agent:
    """
        Generic agent class
        * name: Unique identifier 
        * alpha: Q-learning alpha factor, representing agent's memory
        * gamma: Q-learning gamma factor, representing agent's risk aversity
        * epsilon: e-greedy policy e factor, representing agent's curiosity
        * q_table: Q-learning table
        * size_unk: Total number of cells to explore in q_table
        * proportion_unk: (Estimation of the) proportions of cells yet to explore
        * history: List of all purchases
    """
    
    def __init__(self,
                 alpha: float, 
                 gamma: float, 
                 epsilon: float,
                 name: Union[str, int] = ""):
        
        self.name: str = str(name) if name != "" else str(uuid4())
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        
        # Q-learning table
        self.q_table: np.array = None           # To be set in child class
        self.size_unk: int = None               # To be set in child class
        self.proportion_unk: int = 1
            
        # History of Agent's transactions
        self.history: list = None               # To be set in child class
            
            
            
    def get_q_table(self) -> np.array:
        return self.q_table

    def get_sparsity(self) -> float:
        return np.count_nonzero(self.q_table == 0) / np.count_nonzero(self.q_table != None)
            
    def get_size_unk(self) -> int:
        return self.size_unk
            
    def get_history(self) -> list:
        return self.history
    
    

    def plot_history(self) -> None:
        """ Displays transactions history """
        raise NotImplementedError
        


    def epsilon_updated(self) -> float:
        """ Returns epsilon factor dynamically adjusted """
        # Compute epsilon, given the current state of exploration of the Q-table
        epsilon = update_epsilon(self.proportion_unk, self.epsilon)
        # Update the estimation of the number of cells yet to explore
        self.proportion_unk = max(0, self.proportion_unk - epsilon / self.size_unk)
        return epsilon 


    def learn(self) -> None:
        """ Update Q-table ("learn") and reinitialize params """
        raise NotImplementedError
