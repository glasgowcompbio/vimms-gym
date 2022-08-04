import os
import sys

import pandas as pd

sys.path.append('..')

import streamlit as st
import torch as th
import numpy as np
from stable_baselines3 import PPO, DQN

from experiments import preset_qcb_small, preset_qcb_medium, preset_qcb_large

from vimms_gym.common import METHOD_PPO, METHOD_TOPN, METHOD_DQN
from vimms_gym.env import DDAEnv
from vimms_gym.evaluation import Episode, pick_action

METHOD_DQN_COV = 'DQN (Coverage)'
METHOD_DQN_INT = 'DQN (Intensity)'


class Trajectory():
    def __init__(self, importance):
        self.importance = importance
        self.states = []
        self.actions = []
        self.rewards = []
        self.timesteps = []
        self.flag = False

    def add_pairs(self, pairs: list):
        for pair in pairs:
            self.states.append(pair[0])
            self.actions.append(pair[1])
            self.rewards.append(pair[2])
            self.timesteps.append(pair[3])
        self.flag = True

    def get_df(self, max_peaks, feature_name):
        df = pd.DataFrame({'timestep': self.timesteps, 'action': self.actions, 'reward': self.rewards})
        for name in feature_name:
            feature_vector = []
            for i in range(len(self.actions)):
                if self.actions[i] == max_peaks:
                    feature_vector.append('None')
                elif self.actions[i] != max_peaks:
                    feature_vector.append(self.states[i][name][self.actions[i]])
            df.insert(loc=3, column=name, value=feature_vector)

        return df

class PriorityQueue():
    def __init__(self):
        self.length = 0
        self.items = []
        self.I_values = []
        self.indexes = []

    def insert(self, item, importance, index, budget):
        self.items.append(item)
        self.I_values.append(importance)
        self.indexes.append(index)
        if self.length < budget:
            self.length += 1

    def pop(self):
        del (self.I_values[-1])
        del (self.items[-1])
        del (self.indexes[-1])

    def order(self):
        for i in range(len(self.I_values)):
            for j in range(len(self.I_values) - 1 - i):
                if self.I_values[j] < self.I_values[j + 1]:
                    self.I_values[j], self.I_values[j + 1] = self.I_values[j + 1], self.I_values[j]
                    self.items[j], self.items[j + 1] = self.items[j + 1], self.items[j]
                    self.indexes[j], self.indexes[j + 1] = self.indexes[j + 1], self.indexes[j]

    def get_min_I(self):
        return min(self.I_values)


def get_parameters(preset_name):
    with st.spinner('Extracting distributions from mzML'):
        if preset_name == 'QCB_chems_small':
            return preset_qcb_small()
        elif preset_name == 'QCB_chems_medium':
            return preset_qcb_medium()
        elif preset_name == 'QCB_resimulated_medium':
            return preset_qcb_medium(extract_chromatograms=True)
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

    if method == METHOD_PPO:
        raise ValueError('PPO is no longer supported')

    elif method == METHOD_DQN_COV:
        alpha = 0.75
        in_dir = os.path.abspath(os.path.join('..', 'notebooks', preset, 'results_%.2f' % alpha))
        fname = os.path.join(in_dir, '%s_%s.zip' % (env_name, METHOD_DQN))
        model = DQN.load(fname, custom_objects=custom_objects)
        params['env']['alpha'] = alpha

    elif method == METHOD_DQN_INT:
        alpha = 0.25
        in_dir = os.path.abspath(os.path.join('..', 'notebooks', preset, 'results_%.2f' % alpha))
        fname = os.path.join(in_dir, '%s_%s.zip' % (env_name, METHOD_DQN))
        model = DQN.load(fname, custom_objects=custom_objects)
        params['env']['alpha'] = alpha

    elif method == METHOD_TOPN:
        min_ms1_intensity = 5000
        N = 10  # from optimise_baselines.ipynb
        rt_tol = 15  # from optimise_baselines.ipynb
        params['env']['rt_tol'] = rt_tol

        # FIXME: this makes the Top-N reward not comparable to DQN_COV and DQN_INT,
        #        where 0.25 and 0.75 were used
        params['env']['alpha'] = 0.50

    return N, min_ms1_intensity, model, params


def run_simulation(N, chems, max_peaks, method, min_ms1_intensity, model, params, budget, t_length, statesAfter, intervalSize):
    if method in [METHOD_DQN_COV, METHOD_DQN_INT]:
        method = METHOD_DQN

    T = PriorityQueue()
    t = []
    c = 0
    i = 0
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
                if method == METHOD_DQN:
                    if len(t) == t_length:
                        del (t[0])
                    t.append([obs, int(action), round(reward, 4), episode.num_steps])
                    if c > 0:
                        c -= 1

                    with th.no_grad():
                        obs_tensor, _ = model.q_net.obs_to_tensor(obs)
                        q_values = model.q_net(obs_tensor)
                        importance = round(float(max(q_values[0]) - min(q_values[0])), 4)
                    if intervalSize - c == statesAfter:
                        T.items[T.indexes.index(i)].add_pairs(t)

                    if T.length < budget or importance > T.get_min_I():
                        if c == 0:
                            if T.length == budget:
                                T.pop()
                            i += 1
                            tra = Trajectory(importance)
                            T.insert(tra, importance, i, budget)
                            T.order()
                            c = intervalSize

            if episode.num_steps % 500 == 0:
                st.write('Step\t', episode.num_steps, '\tTotal reward\t',
                         episode.get_total_rewards())

            # if episode is finished, break
            if done:
                msg = f'Episode stored into session: {episode.num_steps} timesteps ' \
                      f'with total reward {episode.get_total_rewards()}'
                st.success(msg)
                if method == METHOD_DQN:
                    if not T.items[T.indexes.index(i)].flag:
                        T.items[T.indexes.index(i)].add_pairs(t)

                # store scan information
                vimms_env = env.vimms_env
                episode.scans = vimms_env.controller.scans

                break
    return episode, env, T


def scan_id_to_scan(scans, scan_id):
    all_scans = scans[1] + scans[2]
    filtered = list(filter(lambda sc: sc.scan_id == scan_id, all_scans))
    assert len(filtered) == 1
    return filtered[0]
