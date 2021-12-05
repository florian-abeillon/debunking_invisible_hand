""" agents/Agent """

from typing import List, Union
from uuid import uuid4

import numpy as np
from src.utils import update_curiosity


class Agent:
    """
        Generic agent class
        * type_agent: Agent type (either 'buyer' or 'seller') 
        * name: Unique identifier 
        * alpha: Q-learning alpha factor, representing agent's memory
        * gamma: Q-learning gamma factor, representing agent's risk aversity
        * epsilon: e-greedy policy e factor, representing agent's utlimate curiosity value
        * curiosity_history: Sequence of values representing agent's curiosity over rounds
        * bandit_steepness: Steepness of change exploration -> exploitation in dynamic adjustment of curiosity
        * bandit_breakpoint: Duration of exploration over exploitation in dynamic adjustment of curiosity
        * q_table: Q-learning table
        * size_unk: Total number of cells to explore in q_table
        * proportion_unk: (Estimation of the) proportions of cells yet to explore
        * history: List of all purchases
    """
    
    def __init__(self, type_agent: str,
                       alpha: float, 
                       gamma: float, 
                       epsilon: float,
                       name: Union[str, int] = ""):
        
        self.type_agent: str = type_agent
        self.name: str = str(name) if name != "" else str(uuid4())
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.curiosity_history: List[float] = []
        self.bandit_steepness: float = None         # To be set in child class
        self.bandit_breakpoint: float = None        # To be set in child class
        
        # Q-learning table
        self.q_table: np.array = None               # To be set in child class
        self.size_unk: int = None                   # To be set in child class
        self.proportion_unk: int = 1
            
        # History of Agent's transactions
        self.history: list = None                   # To be set in child class
            
            
            
    def get_type(self) -> str:
        return self.type_agent
    
    def get_q_table(self) -> np.array:
        return self.q_table

    def get_sparsity(self) -> float:
        return np.count_nonzero(self.q_table == 0) / np.count_nonzero(self.q_table != None)
            
    def get_size_unk(self) -> int:
        return self.size_unk
            
    def get_history(self) -> list:
        return self.history
            
    def get_curiosity(self, **kwargs) -> List[float]:
        return self.curiosity_history
    
    

    def plot_history(self) -> None:
        """ 
            Display transactions history 
        """
        raise NotImplementedError

    def plot_curiosity(self) -> None:
        """ 
            Display curiosity evolution over rounds
        """
        raise NotImplementedError
        


    def threshold_bandit(self) -> float:
        """ 
            Return dynamically adjusted curiosity factor epsilon 
        """
        # Compute epsilon, given the current state of exploration of the Q-table
        curiosity = update_curiosity(self.proportion_unk, self.epsilon, self.bandit_steepness, self.bandit_breakpoint)
        self.curiosity_history.append(curiosity)
        # Update the estimation of the number of cells yet to explore
        if self.proportion_unk > 0:
            self.proportion_unk = max(0, self.proportion_unk - curiosity / self.size_unk)
        return curiosity 


    def learn(self) -> None:
        """ 
            Update Q-table ("learn") and reinitialize params 
        """
        raise NotImplementedError
