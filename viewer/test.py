"""
Created on 01/06/2022 00:04
@author: Liu Ziyan
@E-mail: 2650906L@student.gla.ac.uk
"""
# %%
import os
import sys
sys.path.append('..')
import torch as th
from vimms_gym.chemicals import generate_chemicals
from stable_baselines3 import PPO, DQN
from vimms_gym.common import MAX_OBSERVED_LOG_INTENSITY, MAX_ROI_LENGTH_SECONDS
from vimms_gym.common import METHOD_PPO, METHOD_TOPN, METHOD_DQN
from viewer.experiments import preset_qcb_small, preset_qcb_medium, preset_qcb_large
import pandas as pd
import numpy as np
from vimms_gym.env import DDAEnv
from vimms_gym.evaluation import Episode, pick_action

# %%
class Trajectory():
    def __init__(self, importance):
        pass



# %%
def get_parameters(preset_name):
    if preset_name == 'QCB_chems_small':
        return preset_qcb_small()
    elif preset_name == 'QCB_chems_medium':
        return preset_qcb_medium()
    elif preset_name == 'QCB_chems_large':
        return None, None  # not supported yet


def load_model_and_params(preset, method, params):
    params = dict(params)  # make a copy
    model = None
    N = None
    min_ms1_intensity = None

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    env_name = 'DDAEnv'
    in_dir = os.path.abspath(os.path.join('..', 'notebooks', preset, 'results'))

    if method == METHOD_PPO:
        fname = os.path.join(in_dir, '%s_%s.zip' % (env_name, method))
        model = PPO.load(fname, custom_objects=custom_objects)
    elif method == METHOD_DQN:
        fname = os.path.join(in_dir, '%s_%s.zip' % (env_name, method))
        model = DQN.load(fname, custom_objects=custom_objects)
    elif method == METHOD_TOPN:
        min_ms1_intensity = 5000
        N = 10  # from optimise_baselines.ipynb
        rt_tol = 15  # from optimise_baselines.ipynb
        params['env']['rt_tol'] = rt_tol

    return N, min_ms1_intensity, model, params


def run_simulation(N, chems, max_peaks, method, min_ms1_intensity, model, params):
    env = DDAEnv(max_peaks, params)
    obs = env.reset(chems=chems)
    done = False
    episode = Episode(obs)
    while not done:  # repeat until episode is done

        # select an action depending on the observation and method
        action, action_probs = pick_action(
            method, obs, model, env.features, N, min_ms1_intensity)

        # make one step through the simulation
        obs, reward, done, info = env.step(action)

        # FIXME: seems to slow the simulation a lot!
        # image = env.render(mode='rgb_array')

        # store new episodic information
        if obs is not None:
            episode.add_step_data(action, action_probs, obs, reward, info)

        if episode.num_steps % 500 == 0:
            print('Step\t', episode.num_steps, '\tTotal reward\t',
                  episode.get_total_rewards())

        # if episode is finished, break
        if done:
            msg = f'Episode stored into session: {episode.num_steps} timesteps ' \
                  f'with total reward {episode.get_total_rewards()}'
            print(msg)

            # store scan information
            vimms_env = env.vimms_env
            episode.scans = vimms_env.controller.scans

            break
    return episode, env


def scan_id_to_scan(scans, scan_id):
    all_scans = scans[1] + scans[2]
    filtered = list(filter(lambda sc: sc.scan_id == scan_id, all_scans))
    assert len(filtered) == 1
    return filtered[0]


# %%
preset = ['QCB_chems_small', 'QCB_chems_medium', 'QCB_chems_large']
policy = [METHOD_TOPN, METHOD_PPO, METHOD_DQN]
params, max_peaks = get_parameters(preset[0])

# %%
# generate chemicals following the selected preset
chemical_creator_params = params['chemical_creator']
chems = generate_chemicals(chemical_creator_params)
print("{0} chemicals have been generated".format(str(len(chems))))
# %%
# run simulation to generate an episode
N, min_ms1_intensity, model, params = load_model_and_params(preset[0], policy[2], params)
episode, env = run_simulation(N, chems, max_peaks, policy[2], min_ms1_intensity, model, params)
# %%
obs = env.reset(chems=chems)
with th.no_grad():
    obs_tensor, _ = model.q_net.obs_to_tensor(obs)
    q_values = model.q_net(obs_tensor)
# %%

p = model.get_parameters()
# %%
ms2_frags = [e for e in env.vimms_env.mass_spec.fragmentation_events if e.ms_level == 2]

# %%
index = []
ac = []
re = []
intensities = []
excluded = []
roi_length = []
roi_elapsed_time_since_last_frag = []
roi_intensity_at_last_frag = []
roi_min_intensity_since_last_frag = []
roi_max_intensity_since_last_frag = []
scan_id = []
# extract feature data
for step in range(episode.num_steps):
    if episode.actions[step] != max_peaks and step < episode.num_steps - 1:
        intensities.append(
            episode.observations[step + 1]['intensities'][
                episode.actions[step]] * MAX_OBSERVED_LOG_INTENSITY)
        excluded.append(episode.observations[step + 1]['excluded'][episode.actions[step]] * env.rt_tol)
        roi_length.append(
            episode.observations[step + 1]['roi_length'][
                episode.actions[step]] * MAX_ROI_LENGTH_SECONDS)
        roi_elapsed_time_since_last_frag.append(
            episode.observations[step + 1]['roi_elapsed_time_since_last_frag'][
                episode.actions[step]] * MAX_ROI_LENGTH_SECONDS)
        roi_intensity_at_last_frag.append(
            episode.observations[step + 1]['roi_intensity_at_last_frag'][
                episode.actions[step]] * MAX_OBSERVED_LOG_INTENSITY
        )
        roi_min_intensity_since_last_frag.append(
            episode.observations[step + 1]['roi_min_intensity_since_last_frag'][
                episode.actions[step]] * MAX_OBSERVED_LOG_INTENSITY
        )
        roi_max_intensity_since_last_frag.append(
            episode.observations[step + 1]['roi_max_intensity_since_last_frag'][
                episode.actions[step]] * MAX_OBSERVED_LOG_INTENSITY
        )
        ac.append(episode.actions[step])
        re.append(episode.rewards[step])
        index.append(step)
        scan_id.append(episode.get_step_data(step + 1)['info']['current_scan'].scan_id)
# make dataframe
df = pd.DataFrame(
    {'timestep': index, 'action': ac, 'reward': re, 'intensities': intensities, 'excluded': excluded,
     'roi_length': roi_length, 'roi_elapsed_time_since_last_frag': roi_elapsed_time_since_last_frag,
     'roi_intensity_at_last_frag': roi_intensity_at_last_frag,
     'roi_min_intensity_since_last_frag': roi_min_intensity_since_last_frag,
     'roi_max_intensity_since_last_frag': roi_max_intensity_since_last_frag,
     'scan_id': scan_id})

# %%
dfc = pd.DataFrame({'a': ['one'],
                'c': np.arange(1)})
# %%

# %%
