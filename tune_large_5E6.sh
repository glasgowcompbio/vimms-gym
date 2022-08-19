#!/bin/bash

# intensity models

python training.py --tune_model --alpha 0.25 --beta 0.75 --preset QCB_resimulated_large --model DQN --results tune/QCB_resimulated_large/metric_reward_timesteps_1000000_alpha_0.25_beta_0.75 --timesteps 5E6 --n_trials 30 --n_eval_episodes 5 --eval_freq 1E6 --eval_metric reward

python training.py --tune_model --alpha 0.25 --beta 0.75 --preset QCB_resimulated_large --model PPO --results tune/QCB_resimulated_large/metric_reward_timesteps_1000000_alpha_0.25_beta_0.75 --timesteps 5E6 --n_trials 30 --n_eval_episodes 5 --eval_freq 1E6 --eval_metric reward

python training.py --tune_model --alpha 0.25 --beta 0.75 --preset QCB_resimulated_large --model RecurrentPPO --results tune/QCB_resimulated_large/metric_reward_timesteps_1000000_alpha_0.25_beta_0.75 --timesteps 5E6 --n_trials 30 --n_eval_episodes 5 --eval_freq 1E6 --eval_metric reward