import argparse
import os
import socket
import sys

sys.path.append('.')

from experiments import preset_qcb_small, ENV_QCB_SMALL_GAUSSIAN, ENV_QCB_MEDIUM_GAUSSIAN, \
    ENV_QCB_LARGE_GAUSSIAN, ENV_QCB_SMALL_EXTRACTED, ENV_QCB_MEDIUM_EXTRACTED, \
    ENV_QCB_LARGE_EXTRACTED, preset_qcb_medium, preset_qcb_large

# the import order is important to use all cpu cores
# import numpy as np
import torch

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from vimms_gym.env import DDAEnv
from vimms_gym.common import METHOD_PPO, METHOD_DQN


def train_model(model_name, timesteps, params, max_peaks, in_dir, use_subproc=True):
    assert model_name in [METHOD_PPO, METHOD_DQN]

    num_env = 20
    torch_threads = 1  # Set pytorch num threads to 1 for faster training
    if socket.gethostname() == 'cauchy':  # except on cauchy where we have no gpu, only cpu
        torch_threads = 40
    torch.set_num_threads(torch_threads)

    env = DDAEnv(max_peaks, params)
    check_env(env)
    env_name = 'DDAEnv'
    fname = '%s/%s_%s.zip' % (in_dir, env_name, model_name)

    def make_env(rank, seed=0):
        def _init():
            env = DDAEnv(max_peaks, params)
            env.seed(rank)
            env = Monitor(env)
            return env

        set_random_seed(seed)
        return _init

    if not use_subproc:
        env = DummyVecEnv([make_env(i) for i in range(num_env)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(num_env)])

    tensorboard_log = './%s/%s_%s_tensorboard' % (in_dir, env_name, model_name)
    model_params = params['model']
    if model_name == METHOD_PPO:
        model = PPO('MultiInputPolicy', env, tensorboard_log=tensorboard_log, verbose=2,
                    **model_params)
    elif model_name == METHOD_DQN:
        model = DQN('MultiInputPolicy', env, tensorboard_log=tensorboard_log, verbose=2,
                    **model_params)

    single_save_freq = 10E6
    save_freq = max(single_save_freq // num_env, 1)
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=in_dir,
                                             name_prefix='%s_checkpoint' % model_name)
    model.learn(total_timesteps=timesteps, callback=checkpoint_callback, log_interval=1)
    model.save(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training and parameter optimisation script for ViMMS-Gym')

    # model parameters
    parser.add_argument('--model', choices=[
        METHOD_PPO,
        METHOD_DQN
    ], required=True, type=str, help='Specify model name')

    # environment parameters
    parser.add_argument('--env_preset', choices=[
        ENV_QCB_SMALL_GAUSSIAN,
        ENV_QCB_MEDIUM_GAUSSIAN,
        ENV_QCB_LARGE_GAUSSIAN,
        ENV_QCB_SMALL_EXTRACTED,
        ENV_QCB_MEDIUM_EXTRACTED,
        ENV_QCB_LARGE_EXTRACTED
    ], required=True, type=str, help='Specify environmental preset')
    parser.add_argument('--reward_alpha', default=0.5, type=float,
                        help='Weight parameter in the reward function')

    # other parameters
    parser.add_argument('--results_dir', default=os.path.abspath('notebooks'), type=str,
                        help='Base location to store results')
    parser.add_argument('--optimise', default=False, type=bool,
                        help='Optimise hyper-parameters instead of training')

    args = parser.parse_args()
    model_name = args.model
    alpha = args.reward_alpha
    in_dir = args.results_dir + ('_%.2f' % alpha)

    if args.env_preset == ENV_QCB_SMALL_GAUSSIAN:
        params, max_peaks = preset_qcb_small(model_name, alpha=alpha, extract_chromatograms=False)
    elif args.env_preset == ENV_QCB_MEDIUM_GAUSSIAN:
        params, max_peaks = preset_qcb_medium(model_name, alpha=alpha, extract_chromatograms=False)
    elif args.env_preset == ENV_QCB_LARGE_GAUSSIAN:
        params, max_peaks = preset_qcb_large(model_name, alpha=alpha, extract_chromatograms=False)
    elif args.env_preset == ENV_QCB_SMALL_EXTRACTED:
        params, max_peaks = preset_qcb_small(model_name, alpha=alpha, extract_chromatograms=True)
    elif args.env_preset == ENV_QCB_MEDIUM_EXTRACTED:
        params, max_peaks = preset_qcb_medium(model_name, alpha=alpha, extract_chromatograms=True)
    elif args.env_preset == ENV_QCB_LARGE_EXTRACTED:
        params, max_peaks = preset_qcb_large(model_name, alpha=alpha, extract_chromatograms=True)

    timesteps = params['timesteps']
    train_model(model_name, timesteps, params, max_peaks, in_dir, use_subproc=True)
