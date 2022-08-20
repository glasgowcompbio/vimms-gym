#!/bin/bash

# DQN startup run
python training.py --tune_model --tune_reward --preset QCB_resimulated_medium --model DQN --results tune/QCB_resimulated_medium/metric_f1_timesteps_1000000_alpha_auto_beta_auto --timesteps 1E6 --n_trials 25 --n_eval_episodes 1 --eval_freq 100E6 --eval_metric f1

# DQN actual run
python training.py --tune_model --tune_reward --preset QCB_resimulated_medium --model DQN --results tune/QCB_resimulated_medium/metric_f1_timesteps_1000000_alpha_auto_beta_auto --timesteps 1E6 --n_trials 25 --n_eval_episodes 10 --eval_freq 2E5 --eval_metric f1

# PPO startup run
python training.py --tune_model --tune_reward --preset QCB_resimulated_medium --model PPO --results tune/QCB_resimulated_medium/metric_f1_timesteps_1000000_alpha_auto_beta_auto --timesteps 1E6 --n_trials 25 --n_eval_episodes 1 --eval_freq 100E6 --eval_metric f1

# PPO actual run
python training.py --tune_model --tune_reward --preset QCB_resimulated_medium --model RecurrentPPO --results tune/QCB_resimulated_medium/metric_f1_timesteps_1000000_alpha_auto_beta_auto --timesteps 1E6 --n_trials 25 --n_eval_episodes 1 --eval_freq 100E6 --eval_metric f1

# Recurrent PPO startup run
python training.py --tune_model --tune_reward --preset QCB_resimulated_medium --model PPO --results tune/QCB_resimulated_medium/metric_f1_timesteps_1000000_alpha_auto_beta_auto --timesteps 1E6 --n_trials 25 --n_eval_episodes 10 --eval_freq 2E5 --eval_metric f1

# Recurrent PPO actual run
python training.py --tune_model --tune_reward --preset QCB_resimulated_medium --model RecurrentPPO --results tune/QCB_resimulated_medium/metric_f1_timesteps_1000000_alpha_auto_beta_auto --timesteps 1E6 --n_trials 25 --n_eval_episodes 10 --eval_freq 2E5 --eval_metric f1