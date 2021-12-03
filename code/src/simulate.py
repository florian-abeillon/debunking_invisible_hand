""" src/simulate """

import random as rd
from typing import List, Tuple

from agents import Buyer, Seller

from src.constants import NB_BUYERS, NB_SELLERS


def get_new_agents(nb_sellers: int = NB_SELLERS, nb_buyers: int = NB_BUYERS, Verbose: bool = False) -> Tuple[List[Seller], List[Buyer]]:
    """ Creates lists of sellers/buyers """
    if Verbose:
        print(f"Creating {nb_sellers} sellers..")
    sellers = [ Seller(name=i) for i in range(nb_sellers) ]
    
    if Verbose:
        print(f"Creating {nb_buyers} buyers..")
    buyers = [ Buyer(name=i) for i in range(nb_buyers) ]

    return sellers, buyers


def play_round(sellers: List[Seller], buyers: List[Buyer], Verbose: bool = False) -> Tuple[List[Seller], List[Buyer]]:
    """ Plays a round """
    
    # Consider buyers in a random order
    rd.shuffle(buyers)

    for buyer in buyers:
        if Verbose:
            print(f"Buyer {buyer.name}: budget_left={buyer.budget_left}")
        
        # Get a random number of sellers to present to the buyer
        k = rd.randint(0, NB_SELLERS)
        # TODO
        # k = NB_SELLERS
        idx_sellers = rd.sample(range(NB_SELLERS), k=k)
        
        for idx_seller in idx_sellers:
            
            seller = sellers[idx_seller]
            price, qty = seller.get_price(), seller.get_qty()
            if Verbose:
                print(f"> Seller {seller.name}: price_sell={price}  qty_left={qty}")

            qty_bought = buyer.buy(price, qty)
            if qty_bought > 0:
                sellers[idx_seller].sell(qty_bought)
                if Verbose:
                    print(f">> {qty_bought} goods purchased/sold")
                    print(f">> Buyer's budget_left={buyer.budget_left}")
                    print(f">> Seller's qty_left={sellers[idx_seller].get_qty()}")
                
        if Verbose:
            print()
            
        buyer.learn()
                    
    for seller in sellers:
        seller.learn()
        
    return sellers, buyers
