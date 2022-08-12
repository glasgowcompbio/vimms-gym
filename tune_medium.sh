#!/bin/bash

python training.py --tune_model --alpha 0.25 --beta 0.00 --preset QCB_resimulated_medium --model DQN --results tune/QCB_resimulated_medium/results_alpha_0.25_beta_0.00 --timesteps 5E5 --n_trials 30 --n_eval_episodes 5 --eval_freq 1E5 --eval_metric reward

python training.py --tune_model --alpha 0.75 --beta 0.00 --preset QCB_resimulated_medium --model DQN --results tune/QCB_resimulated_medium/results_alpha_0.75_beta_0.00 --timesteps 5E5 --n_trials 30 --n_eval_episodes 5 --eval_freq 1E5 --eval_metric reward

python training.py --tune_model --alpha 0.50 --beta 0.00 --preset QCB_resimulated_medium --model DQN --results tune/QCB_resimulated_medium/results_alpha_0.50_beta_0.00 --timesteps 5E5 --n_trials 30 --n_eval_episodes 5 --eval_freq 1E5 --eval_metric reward