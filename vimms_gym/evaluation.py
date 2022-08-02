import streamlit as st
import numpy as np
from vimms.Evaluation import evaluate_simulated_env, EvaluationData

from vimms_gym.common import METHOD_RANDOM, METHOD_FULLSCAN, METHOD_TOPN, METHOD_PPO, METHOD_DQN
from vimms_gym.env import DDAEnv
from vimms_gym.policy import random_policy, fullscan_policy, topN_policy, get_ppo_action_probs


class Episode():
    def __init__(self, initial_obs):
        self.env = None
        self.rewards = []
        self.observations = [initial_obs]
        self.actions = []
        self.action_probs = []
        self.infos = []
        self.num_steps = 0
        self.scans = None

    def add_step_data(self, action, action_probs, obs, reward, info):
        self.actions.append(action)
        self.action_probs.append(action_probs)
        self.observations.append(obs)
        self.rewards.append(reward)
        self.infos.append(info)
        self.num_steps += 1

    def get_step_data(self, i):
        return {
            'state': self.observations[i],
            'reward': self.rewards[i],
            'action': self.actions[i],
            'action_prob': self.action_probs[i],
            'info': self.infos[i]
        }

    def evaluate_environment(self, env, intensity_threshold):
        vimms_env = env.vimms_env
        self.eval_data = EvaluationData(vimms_env)
        self.eval_res = evaluate(vimms_env, intensity_threshold)
        return self.eval_res

    def get_total_rewards(self):
        return np.sum(self.rewards)


def evaluate(env, intensity_threshold):
    # env can be either a DDAEnv or a ViMMS' Environment object
    try:
        vimms_env = env.vimms_env
    except AttributeError:
        vimms_env = env

    # call vimms codes to compute various statistics
    vimms_env_res = evaluate_simulated_env(vimms_env)
    count_fragmented = np.count_nonzero(vimms_env_res['times_fragmented'])
    count_ms1 = len(vimms_env.controller.scans[1])
    count_ms2 = len(vimms_env.controller.scans[2])
    ms1_ms2_ratio = float(count_ms1) / count_ms2
    efficiency = float(count_fragmented) / count_ms2

    # get all base chemicals used as input to the mass spec
    all_chems = set(
        chem.get_original_parent() for chem in vimms_env.mass_spec.chemicals
    )

    # assume all base chemicals are unfragmented
    fragmented_intensities = {chem: 0.0 for chem in all_chems}

    # loop through ms2 scans, getting frag_events
    for ms2_scan in vimms_env.controller.scans[2]:
        frag_events = ms2_scan.fragevent
        if frag_events is not None:  # if a chemical has been fragmented ...

            # get the frag event for this scan
            # TODO: assume only 1 chemical has been fragmented
            # works for DDA but not for DIA
            event = frag_events[0]

            # get the base chemical that was fragmented
            base_chem = event.chem.get_original_parent()

            # store the max intensity of fragmentation for this base chem
            parent_intensity = event.parents_intensity[0]
            fragmented_intensities[base_chem] = max(
                parent_intensity, fragmented_intensities[base_chem])

    TP = 0  # chemicals hit correctly (above threshold)
    FP = 0  # chemicals hit incorrectly (below threshold)
    FN = 0  # chemicals not hit
    for chem in fragmented_intensities:
        frag_int = fragmented_intensities[chem]
        if frag_int > 0:  # chemical was fragmented ...
            if fragmented_intensities[chem] > (intensity_threshold * chem.max_intensity):
                TP += 1  # above threshold
            else:
                FP += 1  # below threshold
        else:
            FN += 1  # chemical was not fragmented

    # compute precision, recall, f1
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0.0

    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        precision = 0.0

    try:
        f1 = 2 * (recall * precision) / (recall + precision)
    except ZeroDivisionError:
        f1 = 0.0

    eval_res = {
        'coverage_prop': '%.3f' % vimms_env_res['coverage_proportion'][0],
        'intensity_prop': '%.3f' % vimms_env_res['intensity_proportion'][0],
        'ms1/ms2 ratio': '%.3f' % ms1_ms2_ratio,
        'efficiency': '%.3f' % efficiency,
        'TP': '%d' % TP,
        'FP': '%d' % FP,
        'FN': '%d' % FN,
        'precision': '%.3f' % precision,
        'recall': '%.3f' % recall,
        'f1': '%.3f' % f1
    }
    return eval_res


def run_method(env_name, env_params, max_peaks, chem_list, method, out_dir,
               N=10, min_ms1_intensity=5000, model=None,
               print_eval=False, print_reward=False, mzml_prefix=None,
               intensity_threshold=0.5):
    if method in [METHOD_DQN, METHOD_PPO]:
        assert model is not None

    # to store all results across all loop of chem_list
    all_episodic_results = []

    for i in range(len(chem_list)):
        chems = chem_list[i]
        if print_reward:
            print(f'\nEpisode {i} ({len(chems)} chemicals)')

        env = DDAEnv(max_peaks, env_params)
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

            if print_reward and episode.num_steps % 500 == 0:
                print('steps\t', episode.num_steps, '\ttotal rewards\t', episode.get_total_rewards())

            # if episode is finished, break
            if done:
                break

        if print_reward:
            print(
                f'Finished after {episode.num_steps} timesteps with '
                f'total reward {episode.get_total_rewards()}')

        # save mzML and other info useful for evaluation of the ViMMS environment
        if mzml_prefix is None:
            out_file = '%s_%d.mzML' % (method, i)
        else:
            out_file = '%s_%s_%d.mzML' % (mzml_prefix, method, i)
        env.write_mzML(out_dir, out_file)

        # environment will be evaluated here
        eval_res = episode.evaluate_environment(env, intensity_threshold)
        if print_eval:
            print(eval_res)
        all_episodic_results.append(episode)

    return all_episodic_results


def pick_action(method, obs, model, features, N, min_ms1_intensity):
    action_probs = []

    if method == METHOD_RANDOM:
        action = random_policy(obs)
    elif method == METHOD_FULLSCAN:
        action = fullscan_policy(obs)
    elif method == METHOD_TOPN:
        action = topN_policy(obs, features, N, min_ms1_intensity)
    elif method == METHOD_PPO:
        action, _states = model.predict(obs, deterministic=True)
        action_probs = get_ppo_action_probs(model, obs)
    elif method == METHOD_DQN:
        action, _states = model.predict(obs, deterministic=True)

    return action, action_probs
