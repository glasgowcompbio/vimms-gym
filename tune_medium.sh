#!/bin/bash

# intensity models

python training.py --tune_model --alpha 0.25 --beta 0.00 --preset QCB_resimulated_medium --model PPO --results tune/QCB_resimulated_medium/metric_reward_timesteps_1000000_alpha_0.25_beta_0.00 --timesteps 1E6 --n_trials 50 --n_eval_episodes 5 --eval_freq 2E5 --eval_metric reward

python training.py --tune_model --alpha 0.25 --beta 0.00 --preset QCB_resimulated_medium --model RecurrentPPO --results tune/QCB_resimulated_medium/metric_reward_timesteps_1000000_alpha_0.25_beta_0.00 --timesteps 1E6 --n_trials 50 --n_eval_episodes 5 --eval_freq 2E5 --eval_metric reward

python training.py --tune_model --alpha 0.25 --beta 0.00 --preset QCB_resimulated_medium --model DQN --results tune/QCB_resimulated_medium/metric_reward_timesteps_1000000_alpha_0.25_beta_0.00 --timesteps 1E6 --n_trials 50 --n_eval_episodes 5 --eval_freq 2E5 --eval_metric reward
