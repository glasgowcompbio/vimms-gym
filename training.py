import argparse
import os
import sys
import uuid
import socket
from pprint import pprint

from loguru import logger
from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history, plot_param_importances

sys.path.append('.')

import optuna
from optuna.samplers import TPESampler

# the import order is important to use all cpu cores
# import numpy as np
import torch

from stable_baselines3 import PPO, DQN
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from tune import sample_dqn_params, sample_ppo_params, TrialEvalCallback
from experiments import preset_qcb_small, ENV_QCB_SMALL_GAUSSIAN, ENV_QCB_MEDIUM_GAUSSIAN, \
    ENV_QCB_LARGE_GAUSSIAN, ENV_QCB_SMALL_EXTRACTED, ENV_QCB_MEDIUM_EXTRACTED, \
    ENV_QCB_LARGE_EXTRACTED, preset_qcb_medium, preset_qcb_large

from vimms.Common import create_if_not_exist
from vimms_gym.env import DDAEnv
from vimms_gym.common import METHOD_PPO, METHOD_PPO_RECURRENT, METHOD_DQN, ALPHA, BETA, EVAL_METRIC_REWARD, \
    EVAL_METRIC_F1, EVAL_METRIC_COVERAGE_PROP, EVAL_METRIC_INTENSITY_PROP, \
    EVAL_METRIC_MS1_MS2_RATIO, EVAL_METRIC_EFFICIENCY, GYM_ENV_NAME, GYM_NUM_ENV, USE_SUBPROC

TRAINING_CHECKPOINT_FREQ = 10E6
TRAINING_CHECKPOINT_FREQ = max(TRAINING_CHECKPOINT_FREQ // GYM_NUM_ENV, 1)

EVAL_FREQ = 1E5
N_TRIALS = 100
N_EVAL_EPISODES = 30


def train(model_name, timesteps, params, max_peaks, out_dir, verbose=0):
    set_torch_threads()

    model_params = params['model']
    env = make_environment(max_peaks, params)
    model = init_model(model_name, model_params, env, out_dir=out_dir, verbose=verbose)

    checkpoint_callback = CheckpointCallback(
        save_freq=TRAINING_CHECKPOINT_FREQ, save_path=out_dir,
        name_prefix='%s_checkpoint' % model_name)
    log_interval = 1 if verbose == 2 else 4
    model.learn(total_timesteps=timesteps, callback=checkpoint_callback, log_interval=log_interval)
    fname = '%s/%s_%s.zip' % (out_dir, GYM_ENV_NAME, model_name)
    model.save(fname)


def tune(model_name, timesteps, params, max_peaks, out_dir,
         n_trials, n_eval_episodes, eval_freq, eval_metric, tune_model,
         tune_reward, n_startup_trials=0, verbose=0):
    set_torch_threads()

    # Do not prune before 1/3 of the max budget is used
    n_evaluations = max(1, timesteps // int(eval_freq))
    pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_evaluations // 3)
    logger.info(
        f"Doing {int(n_evaluations)} intermediate evaluations for pruning based on the number of timesteps."
        f" (1 evaluation every {int(eval_freq)} timesteps)"
    )

    # Add stream handler of stdout to show the messages
    study_name = f'{model_name}'
    db_name = os.path.abspath(os.path.join(out_dir, 'study.db'))
    storage_name = f'sqlite:///{db_name}'
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True,
                                pruner=pruner, direction='maximize')
    try:
        objective = Objective(model_name, timesteps, params, max_peaks, out_dir,
                              n_evaluations, n_eval_episodes, eval_metric,
                              tune_model, tune_reward, verbose=verbose)
        study.optimize(objective, n_trials=n_trials)
    except KeyboardInterrupt:
        pass

    trial = study.best_trial
    logger.info('Number of finished trials: ', len(study.trials))
    logger.info('Best trial:')
    logger.info('Value: ', trial.value)
    logger.info('Params: ')
    for key, value in trial.params.items():
        logger.info(f'    {key}: {value}')

    # Write report csv and pickle
    i = 0
    while os.path.exists(os.path.join(out_dir, f'study_{i}.csv')):
        i += 1
    study.trials_dataframe().to_csv(os.path.join(out_dir, f'study_{i}.csv'))

    # Plot optimization result
    try:
        fig1 = plot_optimization_history(study)
        fig1.write_image(os.path.join(out_dir, f'fig1_{i}.png'))
        fig2 = plot_param_importances(study)
        fig2.write_image(os.path.join(out_dir, f'fig2_{i}.png'))
    except (ValueError, ImportError, RuntimeError):
        pass


class Objective(object):
    def __init__(self, model_name, timesteps, params, max_peaks, out_dir,
                 n_evaluations, n_eval_episodes, eval_metric, tune_model, tune_reward,
                 verbose=0):
        self.model_name = model_name
        self.timesteps = timesteps
        self.params = params
        self.max_peaks = max_peaks
        self.out_dir = out_dir
        self.n_evaluations = n_evaluations
        self.n_eval_episodes = n_eval_episodes
        self.eval_metric = eval_metric
        self.tune_model = tune_model
        self.tune_reward = tune_reward
        self.verbose = verbose

    def __call__(self, trial):
        # Sample parameters
        if self.model_name == METHOD_PPO:
            sampled_params = sample_ppo_params(trial, self.tune_model, self.tune_reward)
        elif self.model_name == METHOD_PPO_RECURRENT:
            sampled_params = sample_ppo_params(trial, self.tune_model, self.tune_reward)
        elif self.model_name == METHOD_DQN:
            sampled_params = sample_dqn_params(trial, self.tune_model, self.tune_reward)

        # Generate model and reward parameters
        if self.tune_model:  # if tuning, use the sampled model parameters
            model_params = dict(sampled_params)
            try:
                del model_params['alpha']
                del model_params['beta']
            except KeyError:
                pass
        else:  # otherwise use pre-defined model parameters
            model_params = self.params['model']

        if self.tune_reward:  # if tuning, use the sampled reward parameters
            self.params['env']['alpha'] = sampled_params['alpha']
            self.params['env']['beta'] = sampled_params['beta']
        else:  # otherwise leave them as they are
            pass

        # Create the RL model
        env = make_environment(self.max_peaks, self.params)
        model = init_model(self.model_name, model_params, env, out_dir=self.out_dir,
                           verbose=self.verbose)

        # Create env used for evaluation
        # eval_env = make_environment(self.max_peaks, self.params)
        eval_env = DDAEnv(self.max_peaks, self.params)
        eval_env = Monitor(eval_env)

        # Create the callback that will periodically evaluate
        # and report the performance
        optuna_eval_freq = int(self.timesteps / self.n_evaluations)
        optuna_eval_freq = max(optuna_eval_freq // GYM_NUM_ENV,
                               1)  # adjust for multiple environments
        eval_callback = TrialEvalCallback(
            eval_env, trial, self.eval_metric, best_model_save_path=self.out_dir,
            log_path=self.out_dir,
            n_eval_episodes=self.n_eval_episodes, eval_freq=optuna_eval_freq, deterministic=True,
            verbose=self.verbose
        )

        try:
            log_interval = 1 if self.verbose == 2 else 4
            model.learn(self.timesteps, callback=eval_callback, log_interval=log_interval)
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
            print('Sampled parameters:')
            pprint(sampled_params)
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
    elif model_name == METHOD_PPO_RECURRENT:
        model = RecurrentPPO('MultiInputLstmPolicy', env, tensorboard_log=tensorboard_log, verbose=verbose,
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
    parser.add_argument('--verbose', default=0, type=int,
                        help='Verbosity level')

    # model parameters
    parser.add_argument('--model', choices=[
        METHOD_PPO,
        METHOD_PPO_RECURRENT,
        METHOD_DQN
    ], required=True, type=str, help='Specify model name')
    parser.add_argument('--timesteps', required=True, type=float, help='Training timesteps')
    parser.add_argument('--tune_model', action='store_true',
                        help='Optimise model parameters instead of training')
    parser.add_argument('--tune_reward', action='store_true',
                        help='Optimise reward parameters instead of training')
    parser.add_argument('--n_trials', default=N_TRIALS, type=int,
                        help='How many trials in optuna tuning')
    parser.add_argument('--n_eval_episodes', default=N_EVAL_EPISODES, type=int,
                        help='How many evaluation episodes in optuna tuning')
    parser.add_argument('--eval_freq', default=EVAL_FREQ, type=float,
                        help='Frequency of intermediate evaluation steps before pruning an '
                             'episode in optuna tuning')
    parser.add_argument('--eval_metric', choices=[
        EVAL_METRIC_REWARD,
        EVAL_METRIC_F1,
        EVAL_METRIC_COVERAGE_PROP,
        EVAL_METRIC_INTENSITY_PROP,
        EVAL_METRIC_MS1_MS2_RATIO,
        EVAL_METRIC_EFFICIENCY
    ], type=str, help='Specify evaluation metric in optuna tuning')

    # environment parameters
    parser.add_argument('--preset', choices=[
        ENV_QCB_SMALL_GAUSSIAN,
        ENV_QCB_MEDIUM_GAUSSIAN,
        ENV_QCB_LARGE_GAUSSIAN,
        ENV_QCB_SMALL_EXTRACTED,
        ENV_QCB_MEDIUM_EXTRACTED,
        ENV_QCB_LARGE_EXTRACTED
    ], required=True, type=str, help='Specify environmental preset')
    parser.add_argument('--alpha', default=ALPHA, type=float,
                        help='First weight parameter in the reward function')
    parser.add_argument('--beta', default=BETA, type=float,
                        help='Second weight parameter in the reward function')

    args = parser.parse_args()
    model_name = args.model
    if args.tune_reward:
        alpha = None
        beta = None
    else:
        alpha = args.alpha
        beta = args.beta
    out_dir = os.path.abspath(os.path.join(args.results, args.model))
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
    params, max_peaks = preset_func(model_name, alpha=alpha, beta=beta,
                                    extract_chromatograms=extract)

    # actually train the model here
    if args.tune_model or args.tune_reward:
        tune(model_name, args.timesteps, params, max_peaks, out_dir, args.n_trials,
             args.n_eval_episodes, int(args.eval_freq), args.eval_metric,
             args.tune_model, args.tune_reward, verbose=args.verbose)
    else:
        train(model_name, args.timesteps, params, max_peaks, out_dir, verbose=args.verbose)
