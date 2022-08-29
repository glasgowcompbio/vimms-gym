#!/bin/bash

python training.py --tune_model --tune_reward --preset QCB_resimulated_medium --model DQN --results tune/QCB_resimulated_medium/metric_f1_timesteps_2000000_alpha_auto_beta_auto --timesteps 2E6 --n_trials 30 --n_eval_episodes 30 --eval_freq 4E5 --eval_metric f1

# python training.py --tune_model --tune_reward --preset QCB_resimulated_medium --model PPO --results tune/QCB_resimulated_medium/metric_f1_timesteps_2000000_alpha_auto_beta_auto --timesteps 2E6 --n_trials 30 --n_eval_episodes 30 --eval_freq 4E5 --eval_metric f1

# python training.py --tune_model --tune_reward --preset QCB_resimulated_medium --model RecurrentPPO --results tune/QCB_resimulated_medium/metric_f1_timesteps_2000000_alpha_auto_beta_auto --timesteps 2E6 --n_trials 30 --n_eval_episodes 30 --eval_freq 4E5 --eval_metric f1