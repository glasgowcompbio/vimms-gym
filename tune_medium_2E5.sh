#!/bin/bash

# increasing intensity models

python training.py --tune_model --alpha 0.25 --beta 0.75 --preset QCB_resimulated_medium --model PPO --results tune/QCB_resimulated_medium/metric_reward_timesteps_200000_alpha_0.25_beta_0.75 --timesteps 2E5 --n_trials 50 --n_eval_episodes 5 --eval_freq 1E6 --eval_metric reward

python training.py --tune_model --alpha 0.25 --beta 0.75 --preset QCB_resimulated_medium --model RecurrentPPO --results tune/QCB_resimulated_medium/metric_reward_timesteps_200000_alpha_0.25_beta_0.75 --timesteps 2E5 --n_trials 50 --n_eval_episodes 5 --eval_freq 1E6 --eval_metric reward

python training.py --tune_model --alpha 0.25 --beta 0.75 --preset QCB_resimulated_medium --model DQN --results tune/QCB_resimulated_medium/metric_reward_timesteps_200000_alpha_0.25_beta_0.75 --timesteps 2E5 --n_trials 50 --n_eval_episodes 5 --eval_freq 1E6 --eval_metric reward