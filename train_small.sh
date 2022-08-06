#!/bin/bash

# python training.py --tune --preset QCB_chems_small --model DQN --results tune/QCB_chems_small --alpha 0.25 --timesteps 1E6 --n_trials 100
python training.py --tune --preset QCB_chems_small --model DQN --results tune/QCB_chems_small --alpha 0.50 --timesteps 1E6 --n_trials 100
# python training.py --tune --preset QCB_chems_small --model DQN --results tune/QCB_chems_small --alpha 0.75 --timesteps 1E6 --n_trials 100