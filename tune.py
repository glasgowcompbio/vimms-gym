# modified from:
# - https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py
# - https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/hyperparams_opt.py

import torch.nn as nn
from stable_baselines3.common.callbacks import EvalCallback

from vimms_gym.common import linear_schedule


def sample_dqn_params(trial):
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)
    lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    if lr_schedule == 'linear':
        learning_rate = linear_schedule(learning_rate)

    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 100, 128, 256, 512])
    buffer_size = trial.suggest_categorical('buffer_size',
                                            [int(1e4), int(5e4), int(1e5), int(1e6)])
    exploration_final_eps = trial.suggest_uniform('exploration_final_eps', 0, 0.2)
    exploration_fraction = trial.suggest_uniform('exploration_fraction', 0, 0.5)
    target_update_interval = trial.suggest_categorical('target_update_interval',
                                                       [1, 1000, 5000, 10000, 15000, 20000])
    learning_starts = trial.suggest_categorical('learning_starts', [0, 1000, 5000, 10000, 20000])

    train_freq = trial.suggest_categorical('train_freq', [1, 4, 8, 16, 128, 256, 1000])
    subsample_steps = trial.suggest_categorical('subsample_steps', [1, 2, 4, 8])
    gradient_steps = max(train_freq // subsample_steps, 1)

    net_arch = trial.suggest_categorical('net_arch', ['small', 'medium', 'large'])
    net_arch = {
        'small': [64, 64],
        'medium': [256, 256],
        'large': [512, 512]
    }[net_arch]

    hyperparams = {
        'gamma': gamma,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'buffer_size': buffer_size,
        'train_freq': train_freq,
        'gradient_steps': gradient_steps,
        'exploration_fraction': exploration_fraction,
        'exploration_final_eps': exploration_final_eps,
        'target_update_interval': target_update_interval,
        'learning_starts': learning_starts,
        'policy_kwargs': dict(net_arch=net_arch),
    }

    return hyperparams


def sample_ppo_params(trial):
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical('n_steps', [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)
    lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)
    clip_range = trial.suggest_categorical('clip_range', [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical('n_epochs', [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical('gae_lambda', [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical('max_grad_norm',
                                              [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_uniform('vf_coef', 0, 1)

    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform('log_std_init', -4, 1)

    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical('sde_sample_freq', [-1, 8, 16, 32, 64, 128, 256])

    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])

    # activation_fn = trial.suggest_categorical('activation_fn',
    #                                           ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu'])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == 'linear':
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = trial.suggest_categorical('net_arch', ['small', 'medium', 'large'])
    net_arch = {
        'small': [dict(pi=[64, 64], vf=[64, 64])],
        'medium': [dict(pi=[256, 256], vf=[256, 256])],
        'large': [dict(pi=[512, 512], vf=[512, 512])],
    }[net_arch]

    activation_fn = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'elu': nn.ELU, 'leaky_relu': nn.LeakyReLU}[
        activation_fn]

    return {
        'n_steps': n_steps,
        'batch_size': batch_size,
        'gamma': gamma,
        'learning_rate': learning_rate,
        'ent_coef': ent_coef,
        'clip_range': clip_range,
        'n_epochs': n_epochs,
        'gae_lambda': gae_lambda,
        'max_grad_norm': max_grad_norm,
        'vf_coef': vf_coef,
        # 'sde_sample_freq': sde_sample_freq,
        'policy_kwargs': dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }


class TrialEvalCallback(EvalCallback):
    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=10000, deterministic=True,
                 verbose=0, best_model_save_path=None, log_path=None):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
