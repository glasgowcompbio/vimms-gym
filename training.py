import argparse
import os
import socket
import sys

from vimms.Common import create_if_not_exist

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

GYM_NUM_ENV = 20
GYM_ENV_NAME = 'DDAEnv'
SINGLE_SAVE_FREQ = 10E6


def train_model(model_name, timesteps, params, max_peaks, in_dir, use_subproc=True):
    assert model_name in [METHOD_PPO, METHOD_DQN]

    torch_threads = 1  # Set pytorch num threads to 1 for faster training
    if socket.gethostname() == 'cauchy':  # except on cauchy where we have no gpu, only cpu
        torch_threads = 40
    torch.set_num_threads(torch_threads)

    env = DDAEnv(max_peaks, params)
    check_env(env)
    fname = '%s/%s_%s.zip' % (in_dir, GYM_ENV_NAME, model_name)

    def make_env(rank, seed=0):
        def _init():
            env = DDAEnv(max_peaks, params)
            env.seed(rank)
            env = Monitor(env)
            return env

        set_random_seed(seed)
        return _init

    if not use_subproc:
        env = DummyVecEnv([make_env(i) for i in range(GYM_NUM_ENV)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(GYM_NUM_ENV)])

    tensorboard_log = os.path.join(in_dir, '%s_%s_tensorboard' % (GYM_ENV_NAME, model_name))
    model_params = params['model']
    if model_name == METHOD_PPO:
        model = PPO('MultiInputPolicy', env, tensorboard_log=tensorboard_log, verbose=2,
                    **model_params)
    elif model_name == METHOD_DQN:
        model = DQN('MultiInputPolicy', env, tensorboard_log=tensorboard_log, verbose=2,
                    **model_params)

    save_freq = max(SINGLE_SAVE_FREQ // GYM_NUM_ENV, 1)
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

    # other parameters
    parser.add_argument('--results', default=os.path.abspath('notebooks'), type=str,
                        help='Base location to store results')
    parser.add_argument('--optimise', default=False, type=bool,
                        help='Optimise hyper-parameters instead of training')

    args = parser.parse_args()
    model_name = args.model
    alpha = args.alpha
    in_dir = os.path.abspath(args.results + ('_%.2f' % alpha))
    create_if_not_exist(in_dir)

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
    timesteps = params['timesteps']
    train_model(model_name, timesteps, params, max_peaks, in_dir, use_subproc=True)
