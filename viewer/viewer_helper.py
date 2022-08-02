import os
import sys
sys.path.append('..')

import streamlit as st
import torch as th
import numpy as np
from stable_baselines3 import PPO, DQN

from experiments import preset_qcb_small, preset_qcb_medium, preset_qcb_large

from vimms_gym.common import METHOD_PPO, METHOD_TOPN, METHOD_DQN



sys.path.append('..')
from vimms_gym.env import DDAEnv
from vimms_gym.evaluation import Episode, pick_action


def get_parameters(preset_name):
    with st.spinner('Extracting distributions from mzML'):
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
    with st.spinner('Wait for it...'):
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
                if method == METHOD_PPO:
                    importance = round(np.max(action_probs) - np.min(action_probs), 4)
                if method == METHOD_DQN:
                    with th.no_grad():
                        obs_tensor, _ = model.q_net.obs_to_tensor(obs)
                        q_values = model.q_net(obs_tensor)
                        importance = round(float(max(q_values[0]) - min(q_values[0])), 4)
                if method == METHOD_TOPN:
                    importance = []
                episode.add_step_data(action, action_probs, obs, reward, info, importance)

            if episode.num_steps % 500 == 0:
                st.write('Step\t', episode.num_steps, '\tTotal reward\t',
                         episode.get_total_rewards())

            # if episode is finished, break
            if done:
                msg = f'Episode stored into session: {episode.num_steps} timesteps ' \
                      f'with total reward {episode.get_total_rewards()}'
                st.success(msg)

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
