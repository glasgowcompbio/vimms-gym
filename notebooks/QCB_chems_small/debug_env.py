import sys
from os.path import exists

sys.path.append('../..')

from loguru import logger
import numpy as np
from stable_baselines3.common.env_checker import check_env

from vimms.Common import POSITIVE, load_obj, save_obj
from vimms.ChemicalSamplers import MZMLFormulaSampler, MZMLRTandIntensitySampler, \
    MZMLChromatogramSampler
from vimms.Roi import RoiBuilderParams

from vimms_gym.env import DDAEnv
from vimms_gym.features import obs_to_dfs

n_chemicals = (20, 50)
mz_range = (100, 110)
rt_range = (400, 500)
intensity_range = (1E4, 1E20)

min_mz = mz_range[0]
max_mz = mz_range[1]
min_rt = rt_range[0]
max_rt = rt_range[1]
min_log_intensity = np.log(intensity_range[0])
max_log_intensity = np.log(intensity_range[1])

isolation_window = 0.7
N = 10
rt_tol = 15
mz_tol = 10
min_ms1_intensity = 5000
ionisation_mode = POSITIVE

enable_spike_noise = True
noise_density = 0.1
noise_max_val = 1E3

mzml_filename = '../fullscan_QCB.mzML'
samplers = None
samplers_pickle = 'samplers_QCB_small.p'
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
        'cr_sampler': cr_sampler,
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

max_peaks = 100
env = DDAEnv(max_peaks, params)
check_env(env)

observation = env.reset()
scan_df, count_df = obs_to_dfs(observation, env.features)
print(scan_df)

for i in range(100):
    action = 100
    observation, reward, done, info = env.step(action)
    print(reward, done)
print(scan_df)

action = 0
observation, reward, done, info = env.step(action)
scan_df, count_df = obs_to_dfs(observation, env.features)
print(reward, done)
print(scan_df)

action = 100
observation, reward, done, info = env.step(action)
scan_df, count_df = obs_to_dfs(observation, env.features)
print(reward, done)
print(scan_df)

action = 0
observation, reward, done, info = env.step(action)
scan_df, count_df = obs_to_dfs(observation, env.features)
print(reward, done)
print(scan_df)