
## HOWTO
Install requirements

    pip3 install -r requirements.txt

And then just run 

    python3 main.py

to run a market simulation with the given parameters. Some graphs will be plotted at the end of the simulation.  
You may change parameters in *src/constants.py* and *agents/constants.py* before running *main.py*.


## WHAT
* *agents/* - Market agents code
    * *Agent.py* - Market agent definition
    * *constants.py* - Market agents parameters
    * *utils.py* - Market agents functions
    * *Buyer/* - Buying agent code
        * *Buyer.py* - Buying agent definition
        * *utils.py* - Buying agent specific functions
    * *Seller/* - Selling agent code
        * *Seller.py* - Selling agent definition
        * *utils.py* - Selling agent specific functions
* *display/* - Display functions
    * *display_agents.py* - Display functions for market agents
    * *display_buyers.py* - Display functions for buying agents
    * *display_sellers.py* - Display functions for selling agents
* *src/* - Market environment code
    * *constants.py* - Market environment parameters
    * *simulate.py* - Market environment definition
    * *utils.py* - Market environment functions
* *main.py* - Main function to run market simulation
