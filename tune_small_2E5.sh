#!/bin/bash

# increasing intensity models

python training.py --tune_model --alpha 0.50 --beta 1.00 --preset QCB_chems_small --model DQN --results tune/QCB_chems_small/metric_reward_timesteps_200000_alpha_0.50_beta_1.00 --timesteps 2E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 1E6 --eval_metric reward

python training.py --tune_model --alpha 0.50 --beta 1.00 --preset QCB_chems_small --model PPO --results tune/QCB_chems_small/metric_reward_timesteps_200000_alpha_0.50_beta_1.00 --timesteps 2E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 1E6 --eval_metric reward

python training.py --tune_model --alpha 0.50 --beta 1.00 --preset QCB_chems_small --model RecurrentPPO --results tune/QCB_chems_small/metric_reward_timesteps_200000_alpha_0.50_beta_1.00 --timesteps 2E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 1E6 --eval_metric reward