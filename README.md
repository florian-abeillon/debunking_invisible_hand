# debunking_invisible_hand

> __Debunking the *Invisible Hand*__  
> *~ Les truites saumonées ~*  
> Abeillon, Florian - Arnórsson, Sverrir - Huang, Yilin

## General Introduction

Coding project for ETH Zürich's course *[Agent-Based Modeling and Social System Simulation](https://coss.ethz.ch/education/ABM.html)*.  
This project aims to create a faithful __agent-based representation of a market__ and determine the most important parameters that affect it. 
The market’s model is based on a __Markov Decision Process__ (MDP) adjusted to suite a financial application, while the agents learn through 
updating what they know (ie. results/reward from a particular state). This learning process emulates __Reinforcement Learning__ (RL).
In this way, we want to show that the so-called __*Invisible Hand*__ is not a magical phenomenon, but rather is a consequence of individual-level 
dynamics within a complex system.

## The Model

In our project, we model a market environment where two types of market agents interact. Every agent -- be it a buyer or a seller -- will try and maximize its own reward at every step.  
Agents:
* __Buyer__
    * __Reward__: Number of goods purchased
    * __Parameters__:
        * _Budget_ - Amount of money provided each round
        * _Memory_ - Agent's Q-learning memory ("alpha")
        * _Risk tolerance_ - Agent's Q-learning discount factor ("gamma")
        * _Curiosity_ - Agent's \epsilon-greedy policy factor
        * _Bandit steepness_ - Steepness of change exploration -> exploitation in dynamic adjustment of curiosity over rounds
        * _Bandit breakpoint_ - Duration of exploration over exploitation in dynamic adjustment of curiosity over rounds
        * _Short-sightedness_ - Agent's preference between short- and long-term
        * _Penalty_ - Penalization factor when the agent has some budget leftovers at the end of a round
* __Seller__
    * __Reward__: Profit
    * __Parameters__:
        * _Production price_ - Amount of money required to produce a good
        * _Memory_ - Agent's Q-learning memory ("alpha")
        * _Risk tolerance_ - Agent's Q-learning discount factor ("gamma")
        * _Curiosity_ - Agent's \epsilon-greedy policy factor
        * _Bandit steepness_ - Steepness of change exploration -> exploitation in dynamic adjustment of curiosity over rounds
        * _Bandit breakpoint_ - Duration of exploration over exploitation in dynamic adjustment of curiosity over rounds

-> Note that a _supply function_ may be implemented for Sellers -- rather than a mere production price -- so as to depict economies of scale.  


## Fundamental Questions

Does a simple Agent-Based Modelling of a market lead to the apparition of macro-level behaviours -- an *Invisible Hand*? -- as in *real* markets?  
If so, what are the key parameters, that will drastically change the overall market behaviour?  


## Expected Results

We expected to see some macro-level behaviour, maybe not a clear-cut herd mentality but rather a general tendency of agents to behave in a similar way towards a stable "equilibrium state".


## References 

> Sutton, R., Bach, F., & Barto, A. (2018). Ch. 6 Temporal Difference Learning. MIT Press Ltd.  

> Kvalvaer, M., & Bjerkoy, A. (2019). Replicating financial markets using reinforcement learning; an agent based approach.  

> Huang, C. (2018). Financial trading as a game: A deep reinforcement learning approach. arXiv preprint arXiv:1807.02787.  

> Marco Raberto, Silvano Cincotti, Sergio M. Focardi, & Michele Marchesi (2001). Agent-based simulation of a financial market. Physica A: Statistical Mechanics and its Applications, 299(1), 319-327.  

> LeBaron, B. (2002). Building the Santa Fe artificial stock market. Physica A, 1–20.

## Acknowledgments
Thanks to [Thomas Asikis](https://github.com/asikist-ethz) for its guidance during the project.
