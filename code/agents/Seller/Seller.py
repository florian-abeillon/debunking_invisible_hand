""" agents/Seller/Seller """

import random as rd
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from agents.Agent import Agent
from agents.constants import PRICE_MAX, PRICE_MIN, QTY_MAX, QTY_MIN
from agents.Seller.constants import CURIOSITY, MEMORY, PRICE_PROD
from agents.Seller.utils import get_q_table, get_q_table_size

Q_TABLE = get_q_table(PRICE_MIN, PRICE_MAX, QTY_MIN, QTY_MAX)
Q_TABLE_SIZE = get_q_table_size(PRICE_MIN, PRICE_MAX, QTY_MIN, QTY_MAX)

price_idx = lambda price: price - PRICE_MIN
qty_idx = lambda qty: qty - QTY_MIN


class Seller(Agent):
    """
        Selling agent class
        * name: Unique identifier 
        * alpha: Q-learning alpha factor, representing agent's memory
        * epsilon: e-greedy policy e factor, representing agent's curiosity
        * price_prod: Price of producing one unit of good
        * qty_prod: Quantity of goods produced (to be learned as we go along)
        * price_sell: Price at which the agent will try to sell its goods (to be learned as we go along)
        * qty_left: Quantity of goods that have not been sold (yet)
        * q_table: Q-learning table
        * size_unk: Total number of cells to explore in q_table
        * proportion_unk: (Estimation of the) proportions of cells yet to explore
        * history: List of all sales
    """
    
    def __init__(self,  
                 price_prod: int = PRICE_PROD, 
                 alpha: float = MEMORY, 
                 epsilon: float = CURIOSITY,
                 stochastic: bool = False,
                 name: Union[str, int] = ""):

        super().__init__(alpha, epsilon, name=name)
        
        assert PRICE_MIN <= price_prod <= PRICE_MAX, f"Production price price_prod={price_prod} is not within the price bounds PRICE_MIN={PRICE_MIN} and PRICE_MAX={PRICE_MAX}"
        self.price_prod: int = price_prod
        
        # If stochastic==True, use a gaussian distribution to get price_prod
        if stochastic:
            std_dev = (PRICE_MAX - PRICE_MIN) / 100
            price_prod = rd.gauss(price_prod, std_dev)
            self.price_prod = min(PRICE_MAX, max(PRICE_MIN, int(price_prod)))
            
        # Initialize randomly first selling price
        self.qty_prod: int = rd.randint(QTY_MIN, QTY_MAX)
        self.price_sell: int = rd.randint(self.price_prod, PRICE_MAX)
        
        self.qty_left: int = self.qty_prod
        
        # Q-learning table
        self.q_table: np.array = np.copy(Q_TABLE)
        self.size_unk: int = Q_TABLE_SIZE

        # history is a list of triples ( investment, price, List[qty sold] ) -> one triple for every round
        investment = self.qty_prod * self.price_prod
        self.history: List[Tuple[int, List[int]]] = [ ( investment, self.price_sell, [] ) ]
            
  
        
    def get_price(self) -> int:
        return self.price_sell
            
    def get_qty(self) -> int:
        return self.qty_left
    

    
    def plot_price(self) -> None:
        """ Displays price fluctuations """
        prices = pd.Series([ 
            price for _, price, _ in self.get_history() 
        ])
        prices.plot(
            xlabel="Rounds",
            ylabel="Selling price"
        )
        
    def plot_profit(self) -> None:
        """ Displays profit variations """
        profits = pd.Series([ 
            price * sum(sales) - investment 
            for investment, price, sales in self.get_history() 
        ])
        profits.plot(
            xlabel="Rounds",
            ylabel="Profit"
        )
    
    def plot_history(self) -> None:
        """ Displays sales history (quantity sold over price) """
        df_history = pd.DataFrame([
            ( price, sum(sales) ) for _, price, sales in self.get_history()
        ])
        df_history.plot(
            0, 
            1, 
            kind='scatter', 
            xlim=[ PRICE_MIN, PRICE_MAX ],
            xlabel="Selling price",
            ylabel="Number of sales",
            c=df_history.index, 
            colormap='jet',
            colorbar=True
        )
    
    
    
    def sell(self, qty: int) -> None:
        """ Sell qty goods at self.price_sell """
        assert qty <= self.qty_left, f"Seller {self.name} is trying to sell {qty} goods, but it has only {self.qty_left} left!"
        self.qty_left -= qty
        self.history[-1][2].append(qty)
        
            
    def learn(self) -> None:
        """ Update Q-table ("learn") and reinitialize params """
        
        # Compute reward (profit)
        qty_sold = self.qty_prod - self.qty_left
        reward = qty_sold * self.price_sell - self.qty_prod * self.price_prod
        
        # q_value_before = self.q_table[price_idx(self.price_sell), qty_idx(self.qty_prod)]

        # Update Q-table
        self.q_table[price_idx(self.price_sell), qty_idx(self.qty_prod)] *= 1 - self.alpha
        self.q_table[price_idx(self.price_sell), qty_idx(self.qty_prod)] += self.alpha * reward

        # print(f"Seller {self.name} learning...")
        # print(f"Q-value {self.price_sell, self.qty_prod}: {q_value_before} -> {self.q_table[price_idx(self.price_sell), qty_idx(self.qty_prod)]}")
        # print(f"Reward - {reward}")
        # print("------------------------------------\n")
        
        # Get next price with e-greedy policy
        if rd.random() < self.epsilon_updated():
            # Exploration: Try out a random pair ( price_sell, qty_prod )
            self.price_sell = rd.randint(PRICE_MIN, PRICE_MAX)
            self.qty_prod = rd.randint(QTY_MIN, QTY_MAX)
        else:
            # Exploitation: Go for maximizing pair ( price_sell, qty_prod )
            self.price_sell, self.qty_prod = np.unravel_index(np.argmax(self.q_table), self.q_table.shape)
        
        # Give buyers their budget for next round
        self.qty_left = self.qty_prod        
        # Prepare a new list for next round
        investment = self.qty_prod * self.price_prod
        self.history.append(( investment, self.price_sell, [] ))
