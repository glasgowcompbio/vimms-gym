import argparse
import os
import socket
import sys
import time
from pprint import pprint

from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history, plot_param_importances

sys.path.append('.')

import optuna
from optuna.samplers import TPESampler

# the import order is important to use all cpu cores
# import numpy as np
import torch

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from tune import sample_dqn_params, sample_ppo_params, TrialEvalCallback
from experiments import preset_qcb_small, ENV_QCB_SMALL_GAUSSIAN, ENV_QCB_MEDIUM_GAUSSIAN, \
    ENV_QCB_LARGE_GAUSSIAN, ENV_QCB_SMALL_EXTRACTED, ENV_QCB_MEDIUM_EXTRACTED, \
    ENV_QCB_LARGE_EXTRACTED, preset_qcb_medium, preset_qcb_large

from vimms.Common import create_if_not_exist, save_obj
from vimms_gym.env import DDAEnv
from vimms_gym.common import METHOD_PPO, METHOD_DQN

GYM_ENV_NAME = 'DDAEnv'
if socket.gethostname() == 'cauchy':
    GYM_NUM_ENV = 20
    USE_SUBPROC = True
else:
    GYM_NUM_ENV = 1
    USE_SUBPROC = False
TRAINING_CHECKPOINT_FREQ = 10E6
TRAINING_CHECKPOINT_FREQ = max(TRAINING_CHECKPOINT_FREQ // GYM_NUM_ENV, 1)

MIN_EVAL_FREQ = 1E5
N_EVAL_EPISODES = 5


def train_model(model_name, timesteps, params, max_peaks, out_dir):
    assert model_name in [METHOD_PPO, METHOD_DQN]
    set_torch_threads()

    model_params = params['model']
    env = make_environment(max_peaks, params)
    model = init_model(model_name, model_params, env, out_dir=out_dir, verbose=2)

    checkpoint_callback = CheckpointCallback(
        save_freq=TRAINING_CHECKPOINT_FREQ, save_path=out_dir,
        name_prefix='%s_checkpoint' % model_name)
    model.learn(total_timesteps=timesteps, callback=checkpoint_callback, log_interval=1)
    fname = '%s/%s_%s.zip' % (out_dir, GYM_ENV_NAME, model_name)
    model.save(fname)


def tune_model(model_name, timesteps, params, max_peaks, out_dir, n_trials, n_evaluations,
               n_startup_trials=0):
    assert model_name in [METHOD_PPO, METHOD_DQN]
    set_torch_threads()

    sampler = TPESampler(n_startup_trials=n_startup_trials)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_evaluations // 3)
    print(
        f"Doing {int(n_evaluations)} intermediate evaluations for pruning based on the number of timesteps."
        f" (1 evaluation every {int(MIN_EVAL_FREQ)} timesteps)"
    )

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction='maximize')
    try:
        objective = Objective(model_name, timesteps, params, max_peaks, out_dir, n_evaluations)
        study.optimize(objective, n_trials=n_trials)
    except KeyboardInterrupt:
        pass

    trial = study.best_trial
    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    # create report dir
    report_path = f'report_{GYM_ENV_NAME}_{max_peaks}_{n_trials}_{timesteps}_{int(time.time())}'
    log_dir = os.path.join(out_dir, model_name, report_path)
    create_if_not_exist(log_dir)

    # Write report csv and pickle
    study.trials_dataframe().to_csv(os.path.join(log_dir, 'study.csv'))
    save_obj(study, os.path.join(log_dir, 'study.p'))

    # Plot optimization result
    try:
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)
        fig1.savefig(os.path.join(log_dir, 'fig1.png'))
        fig2.savefig(os.path.join(log_dir, 'fig2.png'))
    except (ValueError, ImportError, RuntimeError):
        pass


class Objective(object):
    def __init__(self, model_name, timesteps, params, max_peaks, out_dir, n_evaluations):
        self.model_name = model_name
        self.timesteps = timesteps
        self.params = params
        self.max_peaks = max_peaks
        self.out_dir = out_dir
        self.n_evaluations = n_evaluations

    def __call__(self, trial):
        # Sample hyperparameters
        if model_name == METHOD_PPO:
            sampled_hyperparams = sample_ppo_params(trial)
        elif model_name == METHOD_DQN:
            sampled_hyperparams = sample_dqn_params(trial)

        # Create the RL model
        env = make_environment(self.max_peaks, self.params)
        model = init_model(self.model_name, sampled_hyperparams, env,
                           out_dir=self.out_dir, verbose=0)

        # Create env used for evaluation
        eval_env = make_environment(self.max_peaks, self.params)

        # Create the callback that will periodically evaluate
        # and report the performance
        optuna_eval_freq = int(self.timesteps / self.n_evaluations)
        optuna_eval_freq = max(optuna_eval_freq // GYM_NUM_ENV,
                               1)  # adjust for multiple environments
        eval_callback = TrialEvalCallback(
            eval_env, trial, best_model_save_path=self.out_dir, log_path=self.out_dir,
            n_eval_episodes=N_EVAL_EPISODES, eval_freq=optuna_eval_freq, deterministic=True
        )

        try:
            model.learn(self.timesteps, callback=eval_callback)
            # Free memory
            model.env.close()
            eval_env.close()
        except (AssertionError, ValueError) as e:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            eval_env.close()
            # Prune hyperparams that generate NaNs
            print(e)
            print('============')
            print('Sampled hyperparams:')
            pprint(sampled_hyperparams)
            raise optuna.exceptions.TrialPruned()
        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward


def init_model(model_name, model_params, env, out_dir=None, verbose=0):
    if out_dir is not None:
        tensorboard_log = os.path.join(out_dir, '%s_%s_tensorboard' % (GYM_ENV_NAME, model_name))

    model = None
    if model_name == METHOD_PPO:
        model = PPO('MultiInputPolicy', env, tensorboard_log=tensorboard_log, verbose=verbose,
                    **model_params)
    elif model_name == METHOD_DQN:
        model = DQN('MultiInputPolicy', env, tensorboard_log=tensorboard_log, verbose=verbose,
                    **model_params)
    assert model is not None
    return model


def set_torch_threads():
    torch_threads = 1  # Set pytorch num threads to 1 for faster training
    if socket.gethostname() == 'cauchy':  # except on cauchy where we have no gpu, only cpu
        torch_threads = 40
    torch.set_num_threads(torch_threads)


def make_environment(max_peaks, params):
    env = DDAEnv(max_peaks, params)
    check_env(env)

    def make_env(rank, seed=0):
        def _init():
            env = DDAEnv(max_peaks, params)
            env.seed(rank)
            env = Monitor(env)
            return env

        set_random_seed(seed)
        return _init

    if not USE_SUBPROC:
        env = DummyVecEnv([make_env(i) for i in range(GYM_NUM_ENV)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(GYM_NUM_ENV)])
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training and parameter optimisation script for ViMMS-Gym')

    parser.add_argument('--results', default=os.path.abspath('notebooks'), type=str,
                        help='Base location to store results')

    # model parameters
    parser.add_argument('--model', choices=[
        METHOD_PPO,
        METHOD_DQN
    ], required=True, type=str, help='Specify model name')
    parser.add_argument('--timesteps', required=True, type=float, help='Training timesteps')
    parser.add_argument('--tune', action='store_true',
                        help='Optimise hyper-parameters instead of training')
    parser.add_argument('--n_trials', default=100, type=int,
                        help='How many trials for optuna tuning')

    # environment parameters
    parser.add_argument('--preset', choices=[
        ENV_QCB_SMALL_GAUSSIAN,
        ENV_QCB_MEDIUM_GAUSSIAN,
        ENV_QCB_LARGE_GAUSSIAN,
        ENV_QCB_SMALL_EXTRACTED,
        ENV_QCB_MEDIUM_EXTRACTED,
        ENV_QCB_LARGE_EXTRACTED
    ], required=True, type=str, help='Specify environmental preset')
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='Weight parameter in the reward function')

    args = parser.parse_args()
    model_name = args.model
    alpha = args.alpha
    out_dir = os.path.abspath(os.path.join(args.results, 'results_%.2f' % alpha))
    create_if_not_exist(out_dir)

    # choose one preset and generate parameters for it
    presets = {
        ENV_QCB_SMALL_GAUSSIAN: {'f': preset_qcb_small, 'extract': False},
        ENV_QCB_MEDIUM_GAUSSIAN: {'f': preset_qcb_medium, 'extract': False},
        ENV_QCB_LARGE_GAUSSIAN: {'f': preset_qcb_large, 'extract': False},
        ENV_QCB_SMALL_EXTRACTED: {'f': preset_qcb_small, 'extract': True},
        ENV_QCB_MEDIUM_EXTRACTED: {'f': preset_qcb_medium, 'extract': True},
        ENV_QCB_LARGE_EXTRACTED: {'f': preset_qcb_large, 'extract': True},
    }
    preset_func = presets[args.preset]['f']
    extract = presets[args.preset]['extract']
    params, max_peaks = preset_func(model_name, alpha=alpha, extract_chromatograms=extract)

    # actually train the model here
    timesteps = args.timesteps
    n_trials = args.n_trials
    if args.tune:
        n_evaluations = max(1, timesteps // int(MIN_EVAL_FREQ))
        tune_model(model_name, timesteps, params, max_peaks, out_dir, n_trials, n_evaluations)
    else:
        train_model(model_name, timesteps, params, max_peaks, out_dir)
