#!/bin/bash

python training.py --tune_model --tune_reward --preset QCB_chems_small --model DQN --results tune/QCB_chems_small/metric_f1_timesteps_250000_alpha_auto_beta_auto --timesteps 2.5E5 --n_trials 30 --n_eval_episodes 30 --eval_freq 5E4 --eval_metric f1

python training.py --tune_model --tune_reward --preset QCB_chems_small --model PPO --results tune/QCB_chems_small/metric_f1_timesteps_250000_alpha_auto_beta_auto --timesteps 2.5E5 --n_trials 30 --n_eval_episodes 30 --eval_freq 5E4 --eval_metric f1

python training.py --tune_model --tune_reward --preset QCB_chems_small --model RecurrentPPO --results tune/QCB_chems_small/metric_f1_timesteps_250000_alpha_auto_beta_auto --timesteps 2.5E5 --n_trials 30 --n_eval_episodes 30 --eval_freq 5E4 --eval_metric f1