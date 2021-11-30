""" agents/Agent """

from typing import Union
from uuid import uuid4

import pandas as pd
import seaborn as sns
from src.constants import PRICE_MAX, PRICE_MIN

from agents.Buyer.constants import CURIOSITY_BUYER, MEMORY_BUYER


class Agent:
    """
        Generic agent class
        * name: Unique identifier 
        * alpha: Q-learning alpha factor, representing agent's memory
        * epsilon: e-greedy policy e factor, representing agent's curiosity
        * q_table: Q-learning table
        * to_explore: Total number of cells to explore in q_table
        * to_explore_yet: (Estimation of the) number of cells yet to explore
        * history: List of all purchases
    """
    
    def __init__(self,
                 alpha: float = MEMORY_BUYER, 
                 epsilon: float = CURIOSITY_BUYER,
                 name: Union[str, int] = ""):
        
        self.name: str = str(name) if name != "" else str(uuid4())
        self.alpha: float = alpha
        self.epsilon: float = epsilon
        
        # Q-learning table
        self.q_table: pd.DataFrame = None       # To be set in child class
        self.to_explore: int = None             # To be set in child class
        self.to_explore_yet: int = None             # To be set in child class
            
        # History of all Agent's transactions
        self.history: list = None               # To be set in child class
            
            
            
    def get_q_table(self) -> pd.DataFrame:
        return self.q_table
            
    def get_history(self) -> list:
        return self.history
    
    
    
    @staticmethod
    def plot_history(df_history: pd.DataFrame, x_label: str, y_label: str) -> None:
        """ Displays purchases history (quantity purchased over price) """
        df_history.plot(
            0, 
            1, 
            kind='scatter', 
            xlim=[ PRICE_MIN, PRICE_MAX ],
            xlabel=x_label,
            ylabel=y_label,
            c=df_history.index, 
            colormap='jet',
            colorbar=True
        )
        
    def plot_q_table(self) -> None:
        """ Displays heatmap of learnt Q-table """
        sns.heatmap(
            self.q_table, 
            cmap='jet_r', 
            cbar=True
        )
        


    def epsilon_updated(self) -> float:
        """ Returns epsilon factor dynamically adjusted """
        # Compute epsilon, given the current state of exploration of the Q-table
        frac_to_explore = self.to_explore_yet / self.to_explore
        epsilon = (1 - self.epsilon) * frac_to_explore + self.epsilon
        # Update the estimation of the number of cells yet to explore
        self.to_explore_yet -= epsilon
        return epsilon 


    def learn(self) -> None:
        """ Update Q-table ("learn") and reinitialize params """
        raise NotImplementedError
