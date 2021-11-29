""" agents/Buyer/Buyer """

import math
import random as rd
from typing import List, Tuple, Union
from uuid import uuid4

import pandas as pd
import seaborn as sns
from agents.Buyer.constants import (BUDGET, CURIOSITY_BUYER, MEMORY_BUYER,
                                    MYOPIA_FACTOR, PENALTY_FACTOR,
                                    RISK_AVERSITY_BUYER)
from src.constants import PRICE_MAX, PRICE_MIN, QTY_MAX, QTY_MIN


class Buyer:
    """
        Buying agent class
        * name: Unique identifier 
        * alpha: Q-learning alpha factor, representing agent's memory
        * gamma: Q-learning gamma factor, representing agent's risk aversity
        * epsilon: e-greedy policy e factor, representing agent's curiosity
        * myopia: Factor representing agent's short-sightedness
        * penalty: Penalization factor when there are some budget leftovers at the end of the round
        * budget: Initial budget when starting round
        * budget_left: Budget left, to be used
        * q_table: Q-learning table
        * history: List of all purchases
    """
    
    def __init__(self,  
                 budget: int = BUDGET, 
                 alpha: float = MEMORY_BUYER, 
                 gamma: float = RISK_AVERSITY_BUYER, 
                 epsilon: float = CURIOSITY_BUYER,
                 myopia: float = MYOPIA_FACTOR,
                 penalty: float = PENALTY_FACTOR,
                 stochastic: bool = False,
                 name: Union[str, int] = ""):
        
        self.name: str = str(name) if name != "" else str(uuid4())
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.myopia: float = myopia
        self.penalty: float = penalty
        
        assert PRICE_MIN <= budget <= PRICE_MAX, f"Budget budget={budget} is not within the price bounds PRICE_MIN={PRICE_MIN} and PRICE_MAX={PRICE_MAX}"
        self.budget: int = budget
        
        # If stochastic==True, use a gaussian distribution to get budget
        # TODO: Try out other distributions
        if stochastic:
            std_dev = (PRICE_MAX - PRICE_MIN) / 100
            budget = rd.gauss(budget, std_dev)
            self.budget = min(PRICE_MAX, max(PRICE_MIN, math.floor(budget)))
            
        self.budget_left = self.budget
        
        # Initialize Q-table with zeros
        ## Indexes are pairs (budget_left, price)
        index_list = [ (i, j) for i in range(0, PRICE_MAX+1) for j in range(PRICE_MIN, PRICE_MAX+1) ]
        self.q_table: pd.DataFrame = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(index_list), 
            columns=range(QTY_MIN, QTY_MAX+1), 
            dtype=float
        ).fillna(0.)
            
        # history is a list of lists (one for every round) of triples ( budget_before, price, quantity )
        self.history: List[List[Tuple[int, int, int]]] = [[]]
            
            
            
    def get_q_table(self) -> pd.DataFrame:
        return self.q_table
            
    def get_history(self, non_zero: bool = False) -> list:
        # To return history of actual purchases (when at least one good has been purchased)
        if non_zero:
            return [
                [
                    purchase_hist for purchase_hist in round_hist
                    if purchase_hist[2] > 0                    
                ]
                for round_hist in self.history
            ]
        return self.history
    
    
    
    def plot_history(self) -> None:
        """ Displays purchases history (quantity purchased over price) """
        df_history = pd.DataFrame([
            ( price, qty ) for _, price, qty in self.history
        ])
        df_history.plot(
            0, 
            1, 
            kind='scatter', 
            xlim=[ PRICE_MIN, PRICE_MAX ],
            xlabel="Price",
            ylabel="Number of purchases",
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
    
    
    
    def buy(self, price: int, qty_left: int) -> int:
        """ Returns the number of goods bought at price price """
        
        # Get quantity to buy with e-greedy policy, given that
        # * quantity shall be lower than qty_left (Buyer cannot buy more goods than available)
        # * quantity * price shall be lower than budget (Buyer cannot buy goods for more than its budget)
        # TODO: Try out other exploration/exploitation policies
        
        qty_max = min(qty_left, self.budget_left // price)
        if rd.random() < self.epsilon:
            # Exploration: Try out a random quantity
            qty_to_buy = rd.choice(list(self.q_table.columns[:qty_max+1]))
        else:
            # Exploitation: Go for maximizing quantity
            qty_to_buy = self.q_table.columns[self.q_table.max()[:qty_max+1].argmax()] 
        
        self.history[-1].append(( self.budget_left, price, qty_to_buy ))
        self.budget_left -= qty_to_buy * price
        return qty_to_buy
        
            
    def learn(self) -> None:
        """ Update Q-table ("learn") and reinitialize params """
        
        # Compute number of goods purchased overall this round
        nb_goods_purchased_after = sum([ qty for _, _, qty in self.history[-1] ])
        nb_purchases = len(self.history[-1])
        
        # Compute penalty for budget not spent
        penalty = self.penalty * self.budget_left
        
        for i, purchase in enumerate(self.history[-1]):
            budget, price, qty = purchase
            
            # Compute reward (nb of goods purchased)
            # Weight more nb of goods purchased 
            myopia_factor = self.myopia**(nb_purchases - i)
            nb_goods_purchased_after -= qty
            reward = qty + myopia_factor * (nb_goods_purchased_after - penalty)
            
            # Get max potential Q-value
            if i < nb_purchases - 1:
                budget_left, _, _ = self.history[-1][i+1]
                # TODO: remove price
                potential_reward = self.q_table.loc[budget_left].max().max()
            else:
                potential_reward = 0
        
            # Update Q-table
            self.q_table.loc[(budget, price), qty] *= 1 - self.alpha
            self.q_table.loc[(budget, price), qty] += self.alpha * (reward + self.gamma * potential_reward)
        
        # Prepare a new list for next round
        self.history.append([])
