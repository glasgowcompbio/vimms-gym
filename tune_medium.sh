#!/bin/bash

python training.py --tune_model --preset QCB_resimulated_medium --model PPO --results tune/QCB_resimulated_medium/metric_reward_timesteps_1.0E6_alpha_0.00_beta_0.00 --alpha 0.00 --beta 0.00 --timesteps 1E6 --n_trials 30 --n_eval_episodes 30 --eval_freq 2E5 --eval_metric reward --horizon 1