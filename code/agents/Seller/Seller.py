""" agents/Seller/Seller """

import random as rd
from typing import List, Tuple, Union

import numpy as np
import seaborn as sns
from agents.Agent import Agent
from agents.constants import (CURIOSITY_SELLER, MEMORY_SELLER, PRICE_MAX,
                              PRICE_MIN, PRICE_PROD, QTY_MAX, QTY_MIN,
                              RISK_TOLERANCE_SELLER)
from agents.Seller.utils import get_q_table, get_q_table_size
from display import plot_q_table
from display.display_sellers import plot_variations
from matplotlib import pyplot as plt

Q_TABLE = get_q_table(PRICE_MIN, PRICE_MAX, QTY_MIN, QTY_MAX)
Q_TABLE_SIZE = get_q_table_size(PRICE_MIN, PRICE_MAX, QTY_MIN, QTY_MAX)

idx_price = lambda price: price - PRICE_MIN
idx_qty = lambda qty: qty - QTY_MIN


class Seller(Agent):
    """
        Selling agent class
        * name: Unique identifier 
        * alpha: Q-learning alpha factor, representing agent's memory
        * gamma: Q-learning gamma factor, representing agent's risk aversity
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
                 alpha: float = MEMORY_SELLER, 
                 gamma: float = RISK_TOLERANCE_SELLER, 
                 epsilon: float = CURIOSITY_SELLER,
                 stochastic: bool = False,
                 name: Union[str, int] = ""):

        super().__init__(alpha, gamma, epsilon, name=name)
        
        assert PRICE_MIN <= price_prod <= PRICE_MAX, f"Production price price_prod={price_prod} is not within the price bounds PRICE_MIN={PRICE_MIN} and PRICE_MAX={PRICE_MAX}"
        self.price_prod: int = price_prod
        
        # If stochastic==True, use a gaussian distribution to get price_prod
        if stochastic:
            std_dev = (PRICE_MAX - PRICE_MIN) / 100
            price_prod = rd.gauss(price_prod, std_dev)
            self.price_prod = min(PRICE_MAX, max(PRICE_MIN, int(price_prod)))
            
        # Initialize randomly first selling price
        self.price_sell: int = rd.randint(PRICE_MIN, PRICE_MAX)
        self.qty_prod: int = rd.randint(QTY_MIN, QTY_MAX)
        self.qty_left: int = self.qty_prod
        
        # Q-learning table
        self.q_table: np.array = np.copy(Q_TABLE)
        self.size_unk: int = Q_TABLE_SIZE

        # history is a list of triples ( qty_prod, price, List[qty sold] ) -> one triple for every round
        self.history: List[Tuple[int, int, List[int]]] = [ ( self.qty_prod, self.price_sell, [] ) ]
            
  
        
    def get_price(self) -> int:
        return self.price_sell
            
    def get_qty(self) -> int:
        return self.qty_left
    

    
    def plot_price(self) -> None:
        """ Display price fluctuations """
        plot_variations(self.get_history(), value='price', price_prod=self.price_prod)
    
    def plot_qty(self) -> None:
        """ Display produced quantity fluctuations """
        plot_variations(self.get_history(), value='qty', price_prod=self.price_prod)
        
    def plot_profit(self) -> None:
        """ Display profit fluctuations """
        plot_variations(self.get_history(), value='profit', price_prod=self.price_prod)
    
    def plot_history(self) -> None:
        """ Display sales history (quantity sold over price) """
        history = self.get_history()
        x_lim = len(history)
        y_lim = 100 / PRICE_MIN

        prices, nb_sales = [], []
        for _, price, sales in history:
            prices.append(price)
            nb_sales.append(sum(sales))

        fig = sns.scatterplot(
            x=prices,
            y=nb_sales,
            hue=list(range(x_lim)),
            palette='jet_r',
            alpha=1 - min(0.8, x_lim / 100000)
        )
        fig_baseline = sns.lineplot(
            x=[ self.price_prod, self.price_prod ], 
            y=[ 0, y_lim ]
        )

        fig.set(
            xlim=[ PRICE_MIN, PRICE_MAX ],
            ylim=[ 0, y_lim ],
            xlabel="Selling price",
            ylabel="Number of sales"
        )
        plt.show()
        
    def plot_q_table(self) -> None:
        """ Display heatmap of learnt Q-table """
        plot_q_table(self.get_q_table())
    
    
    
    def sell(self, qty: int) -> None:
        """ Sell 'qty' goods at 'self.price_sell' """
        assert qty <= self.qty_left, f"Seller {self.name} is trying to sell {qty} goods, but it has only {self.qty_left} left!"
        self.qty_left -= qty
        self.history[-1][2].append(qty)
        
            
    def learn(self, Verbose: bool = False) -> None:
        """ Update Q-table ("learn") and reinitialize params """
        
        # Compute reward (profit)
        qty_sold = self.qty_prod - self.qty_left
        reward = qty_sold * self.price_sell - self.qty_prod * self.price_prod
        potential_reward = 0    # QTY_MAX * (self.price_sell - self.price_prod)       # Does it make sense to add this as a potential reward? It will add this same value every time it goes to this cell
        
        if Verbose:
            q_value_before = self.q_table[idx_price(self.price_sell), idx_qty(self.qty_prod)]

        # Update Q-table
        self.q_table[idx_price(self.price_sell), idx_qty(self.qty_prod)] *= 1 - self.alpha
        self.q_table[idx_price(self.price_sell), idx_qty(self.qty_prod)] += self.alpha * (reward + self.gamma * potential_reward)

        if Verbose:
            print(f"Seller {self.name} learning...")
            print(f"Q-value {self.price_sell, self.qty_prod}: {q_value_before} -> {self.q_table[idx_price(self.price_sell), idx_qty(self.qty_prod)]}")
            print(f"Reward - {reward}")
            print("------------------------------------\n")
        
        # Get next price with e-greedy policy
        if rd.random() < self.epsilon_updated():
            # Exploration: Try out a random pair ( price_sell, qty_prod )
            self.price_sell = rd.randint(PRICE_MIN, PRICE_MAX)
            self.qty_prod = rd.randint(QTY_MIN, QTY_MAX)
        else:
            # Exploitation: Go for maximizing pair ( price_sell, qty_prod )
            self.price_sell, self.qty_prod = np.unravel_index(np.argmax(self.q_table), self.q_table.shape)
            self.price_sell += PRICE_MIN
            self.qty_prod += QTY_MIN
        
        # Give buyers their budget for next round
        self.qty_left = self.qty_prod        
        # Prepare a new list for next round
        self.history.append(( self.qty_prod, self.price_sell, [] ))
