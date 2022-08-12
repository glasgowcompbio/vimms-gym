#!/bin/bash

python training.py --preset QCB_resimulated_medium --model DQN --timesteps 1E6 --results notebooks/QCB_resimulated_medium --alpha 0.25 --beta 0.00
python training.py --preset QCB_resimulated_medium --model DQN --timesteps 1E6 --results notebooks/QCB_resimulated_medium --alpha 0.5 --beta 0.00
python training.py --preset QCB_resimulated_medium --model DQN --timesteps 1E6 --results notebooks/QCB_resimulated_medium --alpha 0.75 --beta 0.00
python training.py --preset QCB_resimulated_medium --model DQN --timesteps 1E6 --results notebooks/QCB_resimulated_medium --alpha 0.25 --beta 0.50