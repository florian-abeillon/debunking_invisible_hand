""" agents/Buyer/Buyer """

import random as rd
from typing import List, Tuple, Union

import numpy as np
from agents.Agent import Agent
from agents.Buyer.utils import get_q_table, get_q_table_size
from agents.constants import (BANDIT_BREAKPOINT_BUYER, BANDIT_STEEPNESS_BUYER,
                              BUDGET, BUDGET_MAX, BUDGET_MIN, CURIOSITY_BUYER,
                              MEMORY_BUYER, MYOPIA_BUYER, PENALTY_BUYER,
                              PRICE_MAX, PRICE_MIN, QTY_MAX,
                              RISK_TOLERANCE_BUYER)
from display.display_agents import plot_curiosity
from display.display_buyers import (plot_demand_curve, plot_history,
                                    plot_sub_q_tables, plot_variations_buyers)

Q_TABLE = get_q_table(BUDGET_MAX, PRICE_MIN, PRICE_MAX, QTY_MAX)
Q_TABLE_SIZE = get_q_table_size(BUDGET_MAX, PRICE_MIN, PRICE_MAX, QTY_MAX)

idx_price = lambda price: price - PRICE_MIN


class Buyer(Agent):
    """
        Buying agent class
        * type_agent: 'buyer'
        * name: Unique identifier 
        * alpha: Q-learning alpha factor, representing agent's memory
        * gamma: Q-learning gamma factor, representing agent's risk aversity
        * epsilon: e-greedy policy e factor, representing agent's utlimate curiosity value
        * curiosity_history: Sequence of values representing agent's curiosity over rounds
        * bandit_steepness: Steepness of change exploration -> exploitation in dynamic adjustment of epsilon
        * bandit_breakpoint: Duration of exploration over exploitation in dynamic adjustment of epsilon
        * myopia: Factor representing agent's short-sightedness
        * penalty: Penalization factor when there are some budget leftovers at the end of the round
        * budget: Initial budget when starting round
        * budget_left: Budget left, to be used
        * q_table: Q-learning table
        * size_unk: Total number of cells to explore in q_table
        * proportion_unk: (Estimation of the) proportions of cells yet to explore
        * history: List of all purchases
    """
    
    def __init__(self, budget: int = BUDGET, 
                       alpha: float = MEMORY_BUYER, 
                       gamma: float = RISK_TOLERANCE_BUYER, 
                       epsilon: float = CURIOSITY_BUYER,
                       bandit_steepness: float = BANDIT_STEEPNESS_BUYER,
                       bandit_breakpoint: float = BANDIT_BREAKPOINT_BUYER,
                       myopia: float = MYOPIA_BUYER,
                       penalty: float = PENALTY_BUYER,
                       stochastic: bool = False,
                       name: Union[str, int] = ""):

        super().__init__('buyer', alpha, gamma, epsilon, name=name)
        self.bandit_steepness = bandit_steepness
        self.bandit_breakpoint = bandit_breakpoint
        self.myopia: float = myopia
        self.penalty: float = penalty
        
        assert BUDGET_MIN <= budget <= BUDGET_MAX, f"Budget budget={budget} is not within the price bounds PRICE_MIN={BUDGET_MIN} and BUDGET_MAX={BUDGET_MAX}"
        self.budget: int = budget
        
        # If stochastic==True, use a gaussian distribution to get budget
        if stochastic:
            std_dev = (BUDGET_MAX - BUDGET_MIN) / 100
            budget = rd.gauss(budget, std_dev)
            self.budget = min(BUDGET_MAX, max(BUDGET_MIN, int(budget)))
            
        self.budget_left = self.budget
        
        # Initialize Q-table
        self.q_table: np.array = np.copy(Q_TABLE)[:self.budget+1]
        self.size_unk: int = Q_TABLE_SIZE
            
        # history is a list of lists (one for every round) of triples ( budget_before, price, quantity )
        self.history: List[List[Tuple[int, int, int]]] = [[]]
            
            
            
    def get_budget(self) -> int:
        return self.budget_left

    def get_nb_purchases(self) -> int:
        return sum([ transac[2] for transac in self.get_history()[-1] ])

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
            
    def get_curiosity(self, step: int = 0) -> List[float]:
        curiosity_history = super().get_curiosity()
        if not step:
            step = round(np.mean([ 
                len(hist_round) for hist_round in self.get_history() 
            ]))
        return [
            curiosity_history[i]
            for i in range(0, len(curiosity_history), step)
        ]
    
    

    def plot_budget(self, non_zero: bool = True) -> None:
        """ 
            Display remaining budget variation over rounds 
        """
        plot_variations_buyers(self.get_history(non_zero=non_zero), 'budget', ymin=BUDGET_MIN, ymax=BUDGET_MAX, agent=self)

    def plot_purchases(self) -> None:
        """  
            Display number of purchases variation over rounds 
        """
        plot_variations_buyers(self.get_history(non_zero=True), 'purchases')
    
    def plot_history(self, non_zero: bool = True) -> None:
        """ 
            Display purchases history (quantity purchased over price), for each budget 
        """
        plot_history(self.get_history(non_zero=non_zero), budget=self.budget)

    def plot_curiosity(self) -> None:
        """ 
            Display curiosity evolution over rounds
        """
        plot_curiosity(self.get_curiosity(), self.get_history(non_zero=False), self.epsilon)
        
    def plot_sub_q_tables(self) -> None:
        """ 
            Display heatmap of learnt Q-table, for each budget 
        """
        plot_sub_q_tables(self.get_q_table())

    def plot_demand_curve(self) -> tuple:
        """ 
            Display demand curve from learnt Q-table, for each budget 
        """
        return plot_demand_curve(self.get_q_table())

    
    
    def buy(self, price: int, 
                  qty_left: int) -> int:
        """ 
            Return the number of goods bought at price price 
        """
        assert price > 0, f"price={price} should be positive (no free item)"
        
        # Get quantity to buy with e-greedy policy, given that
        # * quantity shall be lower than qty_left (Buyer cannot buy more goods than available)
        # * quantity * price shall be lower than budget (Buyer cannot buy goods for more than its budget)
        
        qty_lim = min(qty_left, self.budget_left // price)
        if rd.random() < self.threshold_bandit():
            # Exploration: Try out a random quantity
            qty_to_buy = rd.randint(0, qty_lim)
        else:
            # Exploitation: Go for maximizing quantity
            qty_to_buy = np.argmax(self.q_table[self.budget_left, idx_price(price)][:qty_lim+1])
        
        self.history[-1].append(( self.budget_left, price, qty_to_buy ))
        self.budget_left -= qty_to_buy * price
        return qty_to_buy
        
            
    def learn(self, Verbose: bool = False) -> None:
        """ 
            Update Q-table ('learn') and reinitialize params 
        """

        last_round_hist = self.history[-1]
        
        # Compute number of goods purchased overall this round
        nb_goods_purchased_after = sum([ qty for _, _, qty in last_round_hist ])
        nb_purchases = len(last_round_hist)
        
        # Compute penalty for budget not spent
        global_penalty = self.penalty * self.budget_left
        
        # print(f"Buyer {self.name} learning...")
        for i, purchase in enumerate(last_round_hist):
            budget, price, qty = purchase
            
            # Compute local reward (nb of goods purchased from this seller)
            budget_left = last_round_hist[i+1][0] if i < nb_purchases - 1 else self.budget_left
            local_penalty = self.penalty * budget_left
            local_reward = qty - local_penalty
            # Compute flobal reward (nb of goods purchased at the end of the round)
            nb_goods_purchased_after -= qty
            global_reward = nb_goods_purchased_after - global_penalty
            # Compute total reward
            myopia = self.myopia**(nb_purchases - i)
            reward = (1 - myopia) * local_reward + myopia * global_reward

            # Get max potential Q-value
            potential_reward = np.nanmax(self.q_table[budget_left])

            if Verbose:
                q_value_before = self.q_table[budget, idx_price(price), qty]
        
            # Update Q-table
            self.q_table[budget, idx_price(price), qty] *= 1 - self.alpha
            self.q_table[budget, idx_price(price), qty] += self.alpha * (reward + self.gamma * potential_reward)

            if Verbose:
                print(f"Q-value {budget, price, qty}: {q_value_before} -> {self.q_table[budget, idx_price(price), qty]}")
                print(f"Reward - {reward} | Potential reward - {potential_reward}")
                print("------------------------------------\n")
        
        # Give buyers their budget for next round
        self.budget_left = self.budget
        # Prepare a new list for next round
        self.history.append([])
