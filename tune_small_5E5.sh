#!/bin/bash

### DQN ###

# intensity model
python training.py --tune_model --alpha 0.25 --beta 1.00 --preset QCB_chems_small --model DQN --results tune/QCB_chems_small/metric_reward_timesteps_500000_alpha_0.25_beta_1.00 --timesteps 5E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 5E4 --eval_metric reward

# coverage model
python training.py --tune_model --alpha 0.75 --beta 1.00 --preset QCB_chems_small --model DQN --results tune/QCB_chems_small/metric_reward_timesteps_500000_alpha_1.00_beta_1.00 --timesteps 5E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 5E4 --eval_metric reward

# balanced model
python training.py --tune_model --alpha 0.50 --beta 1.00 --preset QCB_chems_small --model DQN --results tune/QCB_chems_small/metric_reward_timesteps_500000_alpha_0.50_beta_1.00 --timesteps 5E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 5E4 --eval_metric reward

### PPO ###

# intensity model
python training.py --tune_model --alpha 0.25 --beta 1.00 --preset QCB_chems_small --model PPO --results tune/QCB_chems_small/metric_reward_timesteps_500000_alpha_0.25_beta_1.00 --timesteps 5E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 5E4 --eval_metric reward

# coverage model
python training.py --tune_model --alpha 0.75 --beta 1.00 --preset QCB_chems_small --model PPO --results tune/QCB_chems_small/metric_reward_timesteps_500000_alpha_1.00_beta_1.00 --timesteps 5E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 5E4 --eval_metric reward

# balanced model
python training.py --tune_model --alpha 0.50 --beta 1.00 --preset QCB_chems_small --model PPO --results tune/QCB_chems_small/metric_reward_timesteps_500000_alpha_0.50_beta_1.00 --timesteps 5E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 5E4 --eval_metric reward

### Recurrent PPO ###

# intensity model
python training.py --tune_model --alpha 0.25 --beta 1.00 --preset QCB_chems_small --model RecurrentPPO --results tune/QCB_chems_small/metric_reward_timesteps_500000_alpha_0.25_beta_1.00 --timesteps 5E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 5E4 --eval_metric reward

# coverage model
python training.py --tune_model --alpha 0.75 --beta 1.00 --preset QCB_chems_small --model RecurrentPPO --results tune/QCB_chems_small/metric_reward_timesteps_500000_alpha_1.00_beta_1.00 --timesteps 5E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 5E4 --eval_metric reward

# balanced model
python training.py --tune_model --alpha 0.50 --beta 1.00 --preset QCB_chems_small --model RecurrentPPO --results tune/QCB_chems_small/metric_reward_timesteps_500000_alpha_0.50_beta_1.00 --timesteps 5E5 --n_trials 30 --n_eval_episodes 10 --eval_freq 5E4 --eval_metric reward