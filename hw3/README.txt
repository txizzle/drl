See http://rll.berkeley.edu/deeprlcourse/docs/hw3.pdf for instructions

The starter code was based on an implementation of Q-learning for Atari
generously provided by Szymon Sidor from OpenAI.

===

To run DQN on Pong with pixels, run the expected: `python run_dqn_atari.py`
To run DQN on Pong with RAM, adjust the desired learning rate (AKA learning rate multipler) on line 133 of "run_dqn_ram.py" and then run: `python run_dqn_ram.py`

To generate the plots: `python plotting.py`

The data from various trials I ran are in respective folders labeled with their LR and whether they were on RAM or pixels.