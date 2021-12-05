""" src/simulate """

import random as rd
from typing import List, Tuple, Union

from agents.Buyer.Buyer import Buyer
from agents.Seller.Seller import Seller

from src.constants import NB_BUYERS, NB_SELLERS


def get_new_agents(nb_sellers: int = NB_SELLERS, 
                   nb_buyers: int = NB_BUYERS, 
                   Verbose: bool = False) -> Tuple[List[Seller], List[Buyer]]:
    """ 
        Create/initialize lists of sellers/buyers 
    """

    if Verbose:
        print(f"Creating {nb_sellers} sellers..")
    sellers = [ Seller(name=i) for i in range(nb_sellers) ]
    
    if Verbose:
        print(f"Creating {nb_buyers} buyers..")
    buyers = [ Buyer(name=i) for i in range(nb_buyers) ]

    return sellers, buyers


def play_round(sellers: List[Seller], 
               buyers: List[Buyer], 
               nb_to_match: Union[int, float] = -1, 
               Verbose: bool = False) -> Tuple[List[Seller], List[Buyer]]:
    """ 
        Play a round 
    """

    assert nb_to_match != 0, f"nb_to_match={nb_to_match} should be negative (sellers are drawn at random for every buyer) or positive!"
    # If nb_to_match, draw a random nb of sellers for every buyer
    random_draw = nb_to_match < 0
    # If 0 < nb_to_match < 1, consider it as a proportion of NB_SELLERS
    if 0 < nb_to_match < 1:
        nb_to_match *= NB_SELLERS
    
    # Consider buyers in a random order
    rd.shuffle(buyers)

    for buyer in buyers:
        if Verbose:
            print(f"Buyer {buyer.name}: budget_left={buyer.budget_left}")
        
        # Get the sellers that will make offers to buyer
        k = rd.randint(1, NB_SELLERS) if random_draw else nb_to_match
        idx_sellers = rd.sample(range(NB_SELLERS), k=k)
        
        for idx_seller in idx_sellers:
            
            seller = sellers[idx_seller]
            price, qty = seller.get_price(), seller.get_qty()
            if Verbose:
                print(f"> Seller {seller.name}: price_sell={price}  qty_left={qty}")

            # Let buyer choose how many goods it will buy from seller
            qty_bought = buyer.buy(price, qty)
            if qty_bought > 0:
                # If any, let seller sell them
                sellers[idx_seller].sell(qty_bought)
                if Verbose:
                    print(f">> {qty_bought} goods purchased/sold")
                    print(f">> Buyer's budget_left={buyer.budget_left}")
                    print(f">> Seller's qty_left={sellers[idx_seller].get_qty()}")
                
        if Verbose:
            print()
            
        # Make buyer learn
        buyer.learn()

    # Make the sellers learn 
    for seller in sellers:
        seller.learn()
        
    return sellers, buyers
