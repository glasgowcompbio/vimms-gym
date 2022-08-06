#!/bin/bash

# python training.py --tune --preset QCB_chems_large --model DQN --results tune/QCB_chems_large --alpha 0.25 --timesteps 5E6 --n_trials 100
python training.py --tune --preset QCB_chems_large --model DQN --results tune/QCB_chems_large --alpha 0.50 --timesteps 5E6 --n_trials 100
# python training.py --tune --preset QCB_chems_large --model DQN --results tune/QCB_chems_large --alpha 0.75 --timesteps 5E6 --n_trials 100