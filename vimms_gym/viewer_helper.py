import os
import sys

import numpy as np
import streamlit as st
from stable_baselines3 import PPO
from vimms.ChemicalSamplers import UniformRTAndIntensitySampler, GaussianChromatogramSampler, \
    UniformMZFormulaSampler
from vimms.Common import POSITIVE

from vimms_gym.common import METHOD_PPO, METHOD_TOPN

sys.path.append('..')
from vimms_gym.env import DDAEnv
from vimms_gym.evaluation import Episode, pick_action


@st.experimental_memo
def preset_1():
    n_chemicals = (2000, 5000)
    mz_range = (100, 600)
    rt_range = (200, 1000)
    intensity_range = (1E4, 1E10)

    min_mz = mz_range[0]
    max_mz = mz_range[1]
    min_rt = rt_range[0]
    max_rt = rt_range[1]

    min_log_intensity = np.log(intensity_range[0])
    max_log_intensity = np.log(intensity_range[1])

    isolation_window = 0.7
    rt_tol = 120
    mz_tol = 10
    ionisation_mode = POSITIVE
    enable_spike_noise = True
    noise_density = 0.1
    noise_max_val = 1E3

    mz_sampler = UniformMZFormulaSampler(min_mz=min_mz, max_mz=max_mz)
    ri_sampler = UniformRTAndIntensitySampler(min_rt=min_rt, max_rt=max_rt,
                                              min_log_intensity=min_log_intensity,
                                              max_log_intensity=max_log_intensity)
    cr_sampler = GaussianChromatogramSampler()
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
    return params


@st.experimental_memo
def preset_2():
    return None


def load_model_and_params(method, params):
    params = dict(params)  # make a copy
    model = None
    N = None
    min_ms1_intensity = None

    if method == METHOD_PPO:
        # TODO: should be uploaded, rather than hardcoded?
        in_dir = os.path.abspath(os.path.join('..', 'notebooks', 'simulated_chems', 'results'))
        env_name = 'DDAEnv'
        model_name = 'PPO'
        fname = os.path.join(in_dir, '%s_%s.zip' % (env_name, model_name))
        # st.write('Loading model from: ', fname)
        model = load_ppo(fname)

    elif method == METHOD_TOPN:
        min_ms1_intensity = 5000
        N = 20  # from optimise_baselines.ipynb
        rt_tol = 30  # from optimise_baselines.ipynb
        params['env']['rt_tol'] = rt_tol

    return N, min_ms1_intensity, model, params


@st.experimental_singleton
def load_ppo(fname):
    model = PPO.load(fname)
    return model


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
                episode.add_step_data(action, action_probs, obs, reward, info)

            if episode.num_steps % 500 == 0:
                st.write('Step\t', episode.num_steps, '\tTotal reward\t',
                         episode.get_total_rewards())

            # if episode is finished, break
            if done:
                msg = f'Episode stored into session: {episode.num_steps} timesteps ' \
                      f'with total reward {episode.get_total_rewards()}'
                st.success(msg)
                break
    return episode
