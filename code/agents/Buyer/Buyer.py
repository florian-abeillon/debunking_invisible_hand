""" agents/Buyer/Buyer """

import math
import random as rd
from typing import List, Tuple, Union

import pandas as pd
from agents.Agent import Agent
from agents.Buyer.constants import (BUDGET, CURIOSITY, MEMORY, MYOPIA, PENALTY,
                                    RISK_TOLERANCE)
from agents.Buyer.utils import get_q_table, get_q_table_size
from agents.constants import (BUDGET_MAX, BUDGET_MIN, PRICE_MAX, PRICE_MIN,
                              QTY_MAX)
from src.utils import plot_q_table

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
        * size_unk: Total number of cells to explore in q_table
        * proportion_unk: (Estimation of the) proportions of cells yet to explore
        * history: List of all purchases
    """
    
    def __init__(self,  
                 budget: int = BUDGET, 
                 alpha: float = MEMORY, 
                 gamma: float = RISK_TOLERANCE, 
                 epsilon: float = CURIOSITY,
                 myopia: float = MYOPIA,
                 penalty: float = PENALTY,
                 stochastic: bool = False,
                 name: Union[str, int] = ""):

        super().__init__(alpha, epsilon, name=name)
        
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
        self.q_table = self.q_table.drop(range(self.budget+1, BUDGET_MAX+1), axis=1)
        self.size_unk: int = Q_TABLE_SIZE
            
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
    
    
    
    # TODO
    def plot_history(self) -> None:
        """ Displays purchases history (quantity purchased over price) """
        df_history = pd.DataFrame([
            [ ( price, qty ) for _, price, qty in round_hist ]
            for round_hist in self.get_history()
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
        
    def plot_sub_q_tables(self, budgets: Union[int, List[int]] = []) -> None:
        """ Displays heatmap of learnt Q-table, for each budget """
        if budgets:
            if type(budgets) == int:
                budgets = [budgets]
        else:
            budgets = self.get_q_table().columns

        for budget in budgets:
            sub_q_table = self.get_q_table().loc[budget].copy(deep=True)
            sub_q_table.loc[:, 1:100] = sub_q_table.loc[:, 1:100][sub_q_table.loc[:, 1:100]!=0.]
            sub_q_table.dropna(axis=1, how='all')
            plot_q_table(sub_q_table)
    
    
    
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
        
        # print(f"Buyer {self.name} learning...")
        for i, purchase in enumerate(last_round_hist):
            budget, price, qty = purchase
            
            # Compute reward (nb of goods purchased)
            # Weight more nb of goods purchased 
            myopia = self.myopia**(nb_purchases - i)
            nb_goods_purchased_after -= qty
            reward = qty + myopia * (nb_goods_purchased_after - penalty)
            
            # Get max potential Q-value
            budget_left = last_round_hist[i+1][0] if i < nb_purchases - 1 else self.budget_left
            potential_reward = self.q_table.loc[budget_left].max().max()

            # q_value_before = self.q_table.loc[(budget, price), qty]
        
            # Update Q-table
            self.q_table.loc[(budget, price), qty] *= 1 - self.alpha
            self.q_table.loc[(budget, price), qty] += self.alpha * (reward + self.gamma * potential_reward)     # TODO: Incentivize more to buy

            # print(f"Q-value {budget, price, qty}: {q_value_before} -> {self.q_table.loc[(budget, price), qty]}")
            # print(f"Reward - {reward} | Potential reward - {potential_reward}")
            # print("------------------------------------\n")
        
        # Give buyers their budget for next round
        self.budget_left = self.budget
        # Prepare a new list for next round
        self.history.append([])
