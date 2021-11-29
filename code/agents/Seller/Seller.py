""" agents/Seller/Seller """

import math
import random as rd
from typing import List, Tuple, Union
from uuid import uuid4

import pandas as pd
import seaborn as sns
from agents.Seller.constants import CURIOSITY_SELLER, MEMORY_SELLER, PRICE_PROD
from src.constants import PRICE_MAX, PRICE_MIN, QTY_MAX, QTY_MIN


class Seller:
    """
        Selling agent class
        * name: Unique identifier 
        * alpha: Q-learning alpha factor, representing agent's memory
        * epsilon: e-greedy policy e factor, representing agent's curiosity
        * price_prod: Price of producing one unit of good   # TODO: Replace to price function
        * qty_prod: Quantity of goods produced (to be learned as we go along)
        * price_sell: Price at which the agent will try to sell its goods (to be learned as we go along)
        * qty_left: Quantity of goods that have not been sold (yet)
        * q_table: Q-learning table
        * history: List of all sales
    """
    
    def __init__(self,  
                 price_prod: int = PRICE_PROD, 
                 alpha: float = MEMORY_SELLER, 
                 epsilon: float = CURIOSITY_SELLER,
                 stochastic: bool = False,
                 name: Union[str, int] = ""):
        
        self.name: str = str(name) if name != "" else str(uuid4())
        self.alpha: float = alpha
        self.epsilon: float = epsilon
        
        assert PRICE_MIN <= price_prod <= PRICE_MAX, f"Production price price_prod={price_prod} is not within the price bounds PRICE_MIN={PRICE_MIN} and PRICE_MAX={PRICE_MAX}"
        self.price_prod: int = price_prod
        
        # If stochastic==True, use a gaussian distribution to get price_prod
        # TODO: Try out other distributions
        if stochastic:
            std_dev = (PRICE_MAX - PRICE_MIN) / 100
            price_prod = rd.gauss(price_prod, std_dev)
            self.price_prod = min(PRICE_MAX, max(PRICE_MIN, math.floor(price_prod)))
            
        # Initialize randomly first selling price
        # TODO: Try out other initialization strategies
        self.qty_prod: int = rd.randint(0, QTY_MAX)
        self.price_sell: int = rd.randint(self.price_prod, PRICE_MAX)
        
        self.qty_left: int = self.qty_prod
        
        # Initialize Q-table with zeros
        self.q_table: pd.DataFrame = pd.DataFrame(
            index=range(PRICE_MIN, PRICE_MAX+1), 
            columns=range(QTY_MIN, QTY_MAX+1), 
            dtype=float
        ).fillna(0.)
            
        # history is a list of triples ( investment, price, List[qty sold] ) -> one triple for every round
        investment = - self.qty_prod * self.price_prod
        self.history: List[Tuple[int, List[int]]] = [ ( investment, self.price_sell, [] ) ]
            
           
        
    def get_price(self) -> int:
        return self.price_sell
            
    def get_qty(self) -> int:
        return self.qty_left
            
    def get_q_table(self) -> pd.DataFrame:
        return self.q_table
            
    def get_history(self) -> list:
        return self.history
    
    
    
    def plot_price(self) -> None:
        """ Displays price fluctuations """
        prices = pd.Series([ 
            price for _, price, _ in self.get_history() 
        ])
        prices.plot()
        
    def plot_profit(self) -> None:
        """ Displays profit variations """
        profits = pd.Series([ 
            price * sum(sales) - investment 
            for investment, price, sales in self.get_history() 
        ])
        profits.plot()
    
    def plot_history(self) -> None:
        """ Displays sales history (quantity sold over price) """
        df_history = pd.DataFrame([
            ( price, sum(sales) )
            for _, price, sales in self.history
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
        
    def plot_q_table(self) -> None:
        """ Displays heatmap of learnt Q-table """
        sns.heatmap(
            self.q_table, 
            cmap='jet_r', 
            cbar=True
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
        
        # Update Q-table
        self.q_table.loc[self.price_sell, self.qty_prod] *= 1 - self.alpha
        self.q_table.loc[self.price_sell, self.qty_prod] += self.alpha * reward
        
        # Get next price with e-greedy policy
        # TODO: Try out other exploration/exploitation policies
        if rd.random() < self.epsilon:
            # Exploration: Try out a random pair ( price_sell, qty_prod )
            self.price_sell = rd.choice(list(self.q_table.index))
            self.qty_prod = rd.choice(list(self.q_table.columns))
        else:
            # Exploitation: Go for maximizing pair ( price_sell, qty_prod )
            self.price_sell = self.q_table.index[self.q_table.max(axis=1).argmax()]
            self.qty_prod = self.q_table.columns[self.q_table.max().argmax()]
        
        # Prepare a new list for next round
        self.history.append(( - self.qty_prod * self.price_prod, self.price_sell, [] ))
