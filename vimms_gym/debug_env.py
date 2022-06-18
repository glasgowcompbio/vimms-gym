import os
import sys
from os.path import exists

from vimms_gym.env import DDAEnv

sys.path.append('../..')

import numpy as np
from loguru import logger

from stable_baselines3 import PPO, DQN

from vimms.Common import POSITIVE, set_log_level_warning, load_obj, save_obj
from vimms.ChemicalSamplers import GaussianChromatogramSampler, MZMLFormulaSampler, \
    MZMLRTandIntensitySampler, MZMLChromatogramSampler
from vimms.Roi import RoiBuilderParams

from vimms_gym.chemicals import generate_chemicals
from vimms_gym.evaluation import Episode, pick_action
from vimms_gym.common import METHOD_TOPN, METHOD_PPO, METHOD_DQN

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
exclusion_t_0 = 15
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
in_dir = None
n_eval_episodes = 1
deterministic = True
set_log_level_warning()
methods = [
    METHOD_TOPN,
]

chemical_creator_params = params['chemical_creator']
chem_list = []
for i in range(n_eval_episodes):
    print(i)
    chems = generate_chemicals(chemical_creator_params)
    chem_list.append(chems)

out_dir = None
env_name = 'DDAEnv'
model_name = 'PPO'
intensity_threshold = 0.5

topN_N = 20
topN_rt_tol = 30

method_eval_results = {}
for method in methods:

    effective_rt_tol = rt_tol
    copy_params = dict(params)
    copy_params['env']['rt_tol'] = effective_rt_tol

    if method == METHOD_PPO:
        fname = os.path.join(in_dir, '%s_%s.zip' % (env_name, model_name))
        model = PPO.load(fname)
    elif method == METHOD_DQN:
        fname = os.path.join(in_dir, '%s_%s.zip' % (env_name, model_name))
        model = DQN.load(fname)
    else:
        model = None
        if method == METHOD_TOPN:
            N = topN_N
            effective_rt_tol = topN_rt_tol
            copy_params = dict(params)
            copy_params['env']['rt_tol'] = effective_rt_tol

    banner = 'method = %s max_peaks = %d N = %d rt_tol = %d' % (
        method, max_peaks, N, effective_rt_tol)
    print(banner)
    print()

    # to store all results across all loop of chem_list
    all_episodic_results = []

    for i in range(len(chem_list)):
        chems = chem_list[i]

        env = DDAEnv(max_peaks, params)
        obs = env.reset(chems=chems)
        done = False

        # lists to store episodic results
        episode = Episode(obs)
        while not done:  # repeat until episode is done

            # select an action depending on the observation and method
            action, action_probs = pick_action(
                method, obs, model, env.features, N, min_ms1_intensity)

            # make one step through the simulation
            obs, reward, done, info = env.step(action)

            # store new episodic information
            if obs is not None:
                episode.add_step_data(action, action_probs, obs, reward, info)
                if episode.num_steps % 500 == 0:
                    print('steps\t', episode.num_steps, '\ttotal rewards\t',
                      episode.get_total_rewards())

            # if episode is finished, break
            if done:
                print(
                    f'Finished after {episode.num_steps} timesteps with '
                    f'total reward {episode.get_total_rewards()}')
                break
