import sys
from os.path import exists
import socket

sys.path.append('../..')
sys.path.append('..')

# the import order is important to use all cpu cores
import numpy as np
import torch

from loguru import logger
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from vimms.Common import POSITIVE, load_obj, save_obj
from vimms.ChemicalSamplers import MZMLFormulaSampler, MZMLRTandIntensitySampler, \
    MZMLChromatogramSampler, GaussianChromatogramSampler, UniformMZFormulaSampler, \
    UniformRTAndIntensitySampler
from vimms.Roi import RoiBuilderParams

from vimms_gym.env import DDAEnv
from vimms_gym.common import linear_schedule

if __name__ == "__main__":

    n_chemicals = (2000, 5000)
    mz_range = (70, 1000)
    rt_range = (0, 1440)
    intensity_range = (1E4, 1E20)

    min_mz = mz_range[0]
    max_mz = mz_range[1]
    min_rt = rt_range[0]
    max_rt = rt_range[1]
    min_log_intensity = np.log(intensity_range[0])
    max_log_intensity = np.log(intensity_range[1])

    isolation_window = 0.7
    N = 10
    rt_tol = 120
    mz_tol = 10
    min_ms1_intensity = 5000
    ionisation_mode = POSITIVE

    enable_spike_noise = True
    noise_density = 0.1
    noise_max_val = 1E3
    alpha = 0.5

    mzml_filename = '../fullscan_QCB.mzML'
    samplers = None
    samplers_pickle = 'samplers_QCB_large.p'
    if exists(samplers_pickle):
        logger.info('Loaded %s' % samplers_pickle)
        samplers = load_obj(samplers_pickle)
        mz_sampler = samplers['mz']
        ri_sampler = samplers['rt_intensity']
        cr_sampler = samplers['chromatogram']
    else:
        logger.info('Creating samplers from %s' % mzml_filename)
        mz_sampler = MZMLFormulaSampler(mzml_filename, min_mz=min_mz, max_mz=max_mz)
        ri_sampler = MZMLRTandIntensitySampler(mzml_filename, min_rt=min_rt, max_rt=max_rt,
                                               min_log_intensity=min_log_intensity,
                                               max_log_intensity=max_log_intensity)
        roi_params = RoiBuilderParams(min_roi_length=3, at_least_one_point_above=1000)
        cr_sampler = MZMLChromatogramSampler(mzml_filename, roi_params=roi_params)
        samplers = {
            'mz': mz_sampler,
            'rt_intensity': ri_sampler,
            'chromatogram': cr_sampler
        }
        save_obj(samplers, samplers_pickle)

    params = {
        'chemical_creator': {
            'mz_range': mz_range,
            'rt_range': rt_range,
            'intensity_range': intensity_range,
            'n_chemicals': n_chemicals,
            'mz_sampler': mz_sampler,
            'ri_sampler': ri_sampler,
            'cr_sampler': GaussianChromatogramSampler(),
        },
        'noise': {
            'enable_spike_noise': enable_spike_noise,
            'noise_density': noise_density,
            'noise_max_val': noise_max_val,
            'mz_range': mz_range
        },
        'env': {
            'ionisation_mode': ionisation_mode,
            'rt_range': rt_range,
            'isolation_window': isolation_window,
            'mz_tol': mz_tol,
            'rt_tol': rt_tol,
            'alpha': alpha
        }
    }

    max_peaks = 200
    in_dir = 'results'

    if socket.gethostname() == 'cauchy':
        num_env = 20
        ppo_torch_threads = 40
        dqn_torch_threads = 40
        ppo_timesteps = 50E6
        dqn_timesteps = 50E6
        train_ppo = True
        train_dqn = False
        use_subproc = True
        single_save_freq = 5E6
        schedule_learning_rate = True
    else:
        num_env = 20
        ppo_torch_threads = 1
        dqn_torch_threads = 1
        ppo_timesteps = 50E6
        dqn_timesteps = 50E6
        train_ppo = True
        train_dqn = False
        use_subproc = True
        single_save_freq = 5E6
        schedule_learning_rate = True

    save_freq = max(single_save_freq // num_env, 1)


    def make_env(rank, seed=0):
        def _init():
            env = DDAEnv(max_peaks, params)
            env.seed(rank)
            env = Monitor(env)
            return env

        set_random_seed(seed)
        return _init


    env = DDAEnv(max_peaks, params)
    check_env(env)

    env_name = 'DDAEnv'

    ####################################################################
    # Train PPO
    ####################################################################

    # # default parameters
    # learning_rate = 0.0003
    # batch_size = 64
    # n_steps = 2048
    # ent_coef = 0.0
    # gamma = 0.99
    # gae_lambda = 0.95
    # hidden_nodes = 64
    # net_arch = [dict(pi=[hidden_nodes, hidden_nodes], vf=[hidden_nodes, hidden_nodes])]
    # policy_kwargs = dict(net_arch=net_arch)

    # parameter set 1
    if schedule_learning_rate:
        learning_rate = linear_schedule(0.0003, min_value=0.0001)
    else:
        learning_rate = 0.0001
    batch_size = 512
    n_steps = 2048
    ent_coef = 0.001
    gamma = 0.90
    gae_lambda = 0.90
    hidden_nodes = 512
    net_arch = [dict(pi=[hidden_nodes, hidden_nodes], vf=[hidden_nodes, hidden_nodes])]
    policy_kwargs = dict(net_arch=net_arch)

    model_name = 'PPO'
    fname = '%s/%s_%s.zip' % (in_dir, env_name, model_name)

    if train_ppo:
        torch.set_num_threads(ppo_torch_threads)
        checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=in_dir,
                                                 name_prefix='PPO_checkpoint')

        if not use_subproc:
            env = DummyVecEnv([make_env(i) for i in range(num_env)])
        else:
            env = SubprocVecEnv([make_env(i) for i in range(num_env)])

        tensorboard_log = './%s/%s_%s_tensorboard' % (in_dir, env_name, model_name)

        model = PPO('MultiInputPolicy', env,
                    tensorboard_log=tensorboard_log,
                    learning_rate=learning_rate, batch_size=batch_size, n_steps=n_steps,
                    ent_coef=ent_coef, gamma=gamma, gae_lambda=gae_lambda,
                    policy_kwargs=policy_kwargs, verbose=2)
        model.learn(total_timesteps=ppo_timesteps, callback=checkpoint_callback, log_interval=1)
        model.save(fname)

    ####################################################################
    # Train DQN
    ####################################################################

    # # original parameters
    # learning_rate = 0.0001
    # batch_size = 32
    # gamma = 0.99
    # exploration_fraction = 0.1
    # exploration_initial_eps = 1.0
    # exploration_final_eps = 0.05
    # hidden_nodes = 64
    # net_arch = [hidden_nodes, hidden_nodes]
    # policy_kwargs = dict(net_arch=net_arch)

    # modified parameters
    if schedule_learning_rate:
        learning_rate = linear_schedule(0.0003, min_value=0.0001)
    else:
        learning_rate = 0.0001
    batch_size = 512
    gamma = 0.90
    exploration_fraction = 0.25
    exploration_initial_eps = 1.0
    exploration_final_eps = 0.10
    hidden_nodes = 512
    net_arch = [hidden_nodes, hidden_nodes]
    policy_kwargs = dict(net_arch=net_arch)

    model_name = 'DQN'
    fname = '%s/%s_%s.zip' % (in_dir, env_name, model_name)

    if train_dqn:
        torch.set_num_threads(dqn_torch_threads)
        checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=in_dir,
                                                 name_prefix='DQN_checkpoint')

        if not use_subproc:
            env = DummyVecEnv([make_env(i) for i in range(num_env)])
        else:
            env = SubprocVecEnv([make_env(i) for i in range(num_env)])
        tensorboard_log = './%s/%s_%s_tensorboard' % (in_dir, env_name, model_name)

        model = DQN('MultiInputPolicy', env,
                    tensorboard_log=tensorboard_log,
                    learning_rate=learning_rate, batch_size=batch_size, gamma=gamma,
                    exploration_fraction=exploration_fraction,
                    exploration_initial_eps=exploration_initial_eps,
                    exploration_final_eps=exploration_final_eps,
                    policy_kwargs=policy_kwargs, verbose=2)
        model.learn(total_timesteps=dqn_timesteps, callback=checkpoint_callback, log_interval=1)
        model.save(fname)
