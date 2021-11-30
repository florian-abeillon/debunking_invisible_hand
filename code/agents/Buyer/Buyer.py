""" agents/Buyer/Buyer """

import math
import random as rd
from typing import List, Tuple, Union

import pandas as pd
from agents.Agent import Agent
from agents.Buyer.constants import (BUDGET, CURIOSITY_BUYER, MEMORY_BUYER,
                                    MYOPIA_FACTOR, PENALTY_FACTOR,
                                    RISK_AVERSITY_BUYER)
from agents.Buyer.utils import get_q_table, get_q_table_size
from src.constants import BUDGET_MAX, BUDGET_MIN, PRICE_MAX, PRICE_MIN, QTY_MAX

Q_TABLE = get_q_table(BUDGET_MAX, PRICE_MIN, PRICE_MAX, QTY_MAX)
Q_TABLE_SIZE = get_q_table_size(BUDGET_MAX, PRICE_MIN, PRICE_MAX, QTY_MAX)


class Buyer(Agent):
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

        super().__init__(alpha=alpha, epsilon=epsilon, name=name)
        
        self.gamma: float = gamma
        self.myopia: float = myopia
        self.penalty: float = penalty
        
        assert BUDGET_MIN <= budget <= BUDGET_MAX, f"Budget budget={budget} is not within the price bounds PRICE_MIN={BUDGET_MIN} and BUDGET_MAX={BUDGET_MAX}"
        self.budget: int = budget
        
        # If stochastic==True, use a gaussian distribution to get budget
        if stochastic:
            std_dev = (BUDGET_MAX - BUDGET_MIN) / 100
            budget = rd.gauss(budget, std_dev)
            self.budget = min(BUDGET_MAX, max(BUDGET_MIN, math.floor(budget)))
            
        self.budget_left = self.budget
        
        # Initialize Q-table
        self.q_table: pd.DataFrame = Q_TABLE
        self.to_explore: int = Q_TABLE_SIZE
        self.to_explore_yet: int = Q_TABLE_SIZE
            
        # history is a list of lists (one for every round) of triples ( budget_before, price, quantity )
        self.history: List[List[Tuple[int, int, int]]] = [[]]
            
            
            
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
        return super().get_history()
    
    
    
    def plot_history(self) -> None:
        """ Displays purchases history (quantity purchased over price) """
        df_history = pd.DataFrame([
            ( price, qty ) for _, price, qty in self.get_history()
        ])
        super().get_history(df_history, "Price", "Number of purchases")
    
    
    
    def buy(self, price: int, qty_left: int) -> int:
        """ Returns the number of goods bought at price price """
        
        # Get quantity to buy with e-greedy policy, given that
        # * quantity shall be lower than qty_left (Buyer cannot buy more goods than available)
        # * quantity * price shall be lower than budget (Buyer cannot buy goods for more than its budget)
        
        qty_max = min(qty_left, self.budget_left // price)
        if rd.random() < self.epsilon_updated():
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

        last_round_hist = self.history[-1]
        
        # Compute number of goods purchased overall this round
        nb_goods_purchased_after = sum([ qty for _, _, qty in last_round_hist ])
        nb_purchases = len(last_round_hist)
        
        # Compute penalty for budget not spent
        penalty = self.penalty * self.budget_left
        
        for i, purchase in enumerate(last_round_hist):
            budget, price, qty = purchase
            
            # Compute reward (nb of goods purchased)
            # Weight more nb of goods purchased 
            myopia_factor = self.myopia**(nb_purchases - i)
            nb_goods_purchased_after -= qty
            reward = qty + myopia_factor * (nb_goods_purchased_after - penalty)
            
            # Get max potential Q-value
            budget_left = last_round_hist[i+1][0] if i < nb_purchases - 1 else self.budget_left
            potential_reward = self.q_table.loc[budget_left].max().max()
        
            # Update Q-table
            self.q_table.loc[(budget, price), qty] *= 1 - self.alpha
            self.q_table.loc[(budget, price), qty] += self.alpha * (reward + self.gamma * potential_reward)
        
        # Prepare a new list for next round
        self.history.append([])
