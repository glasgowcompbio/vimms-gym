#!/bin/bash

python training.py --preset QCB_chems_small --model DQN --timesteps 1E5 --results notebooks/QCB_chems_small --alpha 0.25 --beta 0.00
python training.py --preset QCB_chems_small --model DQN --timesteps 1E5 --results notebooks/QCB_chems_small --alpha 0.5 --beta 0.00
python training.py --preset QCB_chems_small --model DQN --timesteps 1E5 --results notebooks/QCB_chems_small --alpha 0.75 --beta 0.00
python training.py --preset QCB_chems_small --model DQN --timesteps 1E5 --results notebooks/QCB_chems_small --alpha 0.25 --beta 0.50