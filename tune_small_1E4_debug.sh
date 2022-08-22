#!/bin/bash

### DQN ###

# intensity model
python training.py --tune_model --alpha 0.25 --beta 0.75 --preset QCB_chems_small --model DQN --results tune/QCB_chems_small/metric_reward_timesteps_10000_alpha_0.25_beta_0.75 --timesteps 1E4 --n_trials 1 --n_eval_episodes 10 --eval_freq 1E6 --eval_metric reward --verbose 2