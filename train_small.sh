#!/bin/bash

python training.py --preset QCB_chems_small --model DQN --timesteps 2E6 --results notebooks/QCB_chems_small --alpha 0.25 --beta 0.10 --verbose 2
python training.py --preset QCB_chems_small --model DQN --timesteps 2E6 --results notebooks/QCB_chems_small --alpha 0.5 --beta 0.10 --verbose 2
python training.py --preset QCB_chems_small --model DQN --timesteps 2E6 --results notebooks/QCB_chems_small --alpha 0.75 --beta 0.10 --verbose 2