#!/bin/bash

python training.py --tune_model --preset QCB_resimulated_medium --model DQN --results tune/QCB_resimulated_medium/metric_reward_timesteps_1.0E6_alpha_0.50_beta_0.50 --alpha 0.50 --beta 0.50 --timesteps 1E6 --n_trials 30 --n_eval_episodes 30 --eval_freq 2E5 --eval_metric reward --horizon 4

python training.py --tune_model --preset QCB_resimulated_medium --model DQN --results tune/QCB_resimulated_medium/metric_reward_timesteps_1.0E6_alpha_0.25_beta_0.50 --alpha 0.25 --beta 0.50 --timesteps 1E6 --n_trials 30 --n_eval_episodes 30 --eval_freq 2E5 --eval_metric reward --horizon 4

python training.py --tune_model --preset QCB_resimulated_medium --model DQN --results tune/QCB_resimulated_medium/metric_reward_timesteps_1.0E6_alpha_0.75_beta_0.50 --alpha 0.75 --beta 0.50 --timesteps 1E6 --n_trials 30 --n_eval_episodes 30 --eval_freq 2E5 --eval_metric reward --horizon 4
