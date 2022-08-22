#!/bin/bash

### DQN ###

# intensity model
python training.py --tune_model --alpha 0.25 --beta 0.75 --preset QCB_chems_small --model DQN --results tune/QCB_chems_small/metric_reward_timesteps_200000_alpha_0.25_beta_0.75 --timesteps 2E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 1E6 --eval_metric reward

# coverage model
python training.py --tune_model --alpha 0.75 --beta 0.75 --preset QCB_chems_small --model DQN --results tune/QCB_chems_small/metric_reward_timesteps_200000_alpha_0.75_beta_0.75 --timesteps 2E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 1E6 --eval_metric reward

# balanced model
python training.py --tune_model --alpha 0.50 --beta 0.75 --preset QCB_chems_small --model DQN --results tune/QCB_chems_small/metric_reward_timesteps_200000_alpha_0.50_beta_0.75 --timesteps 2E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 1E6 --eval_metric reward

### PPO ###

# intensity model
python training.py --tune_model --alpha 0.25 --beta 0.75 --preset QCB_chems_small --model PPO --results tune/QCB_chems_small/metric_reward_timesteps_200000_alpha_0.25_beta_0.75 --timesteps 2E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 1E6 --eval_metric reward

# coverage model
python training.py --tune_model --alpha 0.75 --beta 0.75 --preset QCB_chems_small --model PPO --results tune/QCB_chems_small/metric_reward_timesteps_200000_alpha_0.75_beta_0.75 --timesteps 2E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 1E6 --eval_metric reward

# balanced model
python training.py --tune_model --alpha 0.50 --beta 0.75 --preset QCB_chems_small --model PPO --results tune/QCB_chems_small/metric_reward_timesteps_200000_alpha_0.50_beta_0.75 --timesteps 2E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 1E6 --eval_metric reward

### Recurrent PPO ###

# intensity model
python training.py --tune_model --alpha 0.25 --beta 0.75 --preset QCB_chems_small --model RecurrentPPO --results tune/QCB_chems_small/metric_reward_timesteps_200000_alpha_0.25_beta_0.75 --timesteps 2E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 1E6 --eval_metric reward

# coverage model
python training.py --tune_model --alpha 0.75 --beta 0.75 --preset QCB_chems_small --model RecurrentPPO --results tune/QCB_chems_small/metric_reward_timesteps_200000_alpha_0.75_beta_0.75 --timesteps 2E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 1E6 --eval_metric reward

# balanced model
python training.py --tune_model --alpha 0.50 --beta 0.75 --preset QCB_chems_small --model RecurrentPPO --results tune/QCB_chems_small/metric_reward_timesteps_200000_alpha_0.50_beta_0.75 --timesteps 2E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 1E6 --eval_metric reward