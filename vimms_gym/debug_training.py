import sys
from os.path import exists

sys.path.append('../..')

import numpy as np
from loguru import logger

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from vimms.Common import POSITIVE, load_obj, save_obj
from vimms.ChemicalSamplers import MZMLFormulaSampler, MZMLRTandIntensitySampler, \
    MZMLChromatogramSampler, GaussianChromatogramSampler
from vimms.Roi import RoiBuilderParams

from vimms_gym.env import DDAEnv

n_chemicals = (2000, 2000)
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

mzml_filename = '../notebooks/fullscan_QCB.mzML'
samplers = None
samplers_pickle = '../notebooks/samplers_fullscan_QCB.mzML.p'
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
    }
}

max_peaks = 200
env = DDAEnv(max_peaks, params)
env_name = 'DDAEnv'

# modified parameters
learning_rate = 0.0001
batch_size = 512
gamma = 0.90
exploration_fraction = 0.25
exploration_initial_eps = 1.0
exploration_final_eps = 0.10
hidden_nodes = 512
total_timesteps = 2000

net_arch = [hidden_nodes, hidden_nodes]
policy_kwargs = dict(net_arch=net_arch)

env = DDAEnv(max_peaks, params)
env = Monitor(env)
env = DummyVecEnv([lambda: env])

model = DQN('MultiInputPolicy', env,
            learning_rate=learning_rate, batch_size=batch_size, gamma=gamma,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            policy_kwargs=policy_kwargs, verbose=2)
model.learn(total_timesteps=total_timesteps, log_interval=1)
