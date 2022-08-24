import os, sys

from vimms_gym.wrappers import HistoryWrapperObsDict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
import pandas as pd

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from vimms.Evaluation import EvaluationData
from vimms_gym.env import DDAEnv
from vimms_gym.common import EVAL_METRIC_REWARD, HISTORY_HORIZON
from vimms_gym.evaluation import evaluate

from tune import TrialEvalCallback
from experiments import preset_qcb_small

def debug_run(fname, max_peaks, params, n_eval_episodes=1, deterministic=True):

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    model = DQN.load(fname, custom_objects=custom_objects)

    eval_env = DDAEnv(max_peaks, params)
    eval_env = HistoryWrapperObsDict(eval_env, horizon=HISTORY_HORIZON)    
    print(eval_env.env_params)

    # wrap env in Monitor, create the trial callback
    eval_env = Monitor(eval_env)
    eval_metric = EVAL_METRIC_REWARD
    eval_callback = TrialEvalCallback(eval_env, None, eval_metric)
    env = eval_callback.eval_env

    assert eval_callback.deterministic == True

    # actual evaluation starts here
    episode_count = 0
    episode_count_target = n_eval_episodes
    current_reward = 0
    current_length = 0
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    episode_starts

    episode_rewards = []
    episode_eval_results = []
    episode_lengths = []
    start = timer()
    while episode_count < episode_count_target:
        actions, states = model.predict(observations, state=states,
                                        episode_start=episode_starts,
                                        deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        # print(actions, rewards, current_reward, current_length, dones)
        episode_starts = dones
        current_reward += rewards[0]
        current_length += 1

        if dones[0]:  # when done, episode would be reset automatically
            val = current_reward
            eval_res = evaluate(eval_data, format_output=False)
            episode_eval_results.append(eval_res)
            end = timer()
            print('Evaluation episode %d finished: metric %f, timedelta=%s' % (
                episode_count, val, str(timedelta(seconds=end - start))))
            start = timer()
            episode_rewards.append(val)
            episode_lengths.append(current_length)
            episode_count += 1
            current_reward = 0
            current_length = 0

        # store previous results for evaluation before 'done'
        # this needs to be here, because VecEnv is automatically reset when done
        inner_env = env.envs[0].env
        eval_data = EvaluationData(inner_env.vimms_env)

    return episode_rewards, episode_eval_results


def eval_res_to_df(rewards, eval_res):
    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)

    metric = [e['coverage_prop'] for e in eval_res]
    coverage_mean = np.mean(metric)
    coverage_std = np.std(metric)

    metric = [e['intensity_prop'] for e in eval_res]
    intensity_prop_mean = np.mean(metric)
    intensity_prop_std = np.std(metric)

    metric = [e['f1'] for e in eval_res]
    f1_mean = np.mean(metric)
    f1_std = np.std(metric)

    results = []
    results.append(['reward', reward_mean, reward_std])
    results.append(['coverage_prop', coverage_mean, coverage_std])
    results.append(['intensity_prop', intensity_prop_mean, intensity_prop_std])
    results.append(['f1', f1_mean, f1_std])
    df = pd.DataFrame(results, columns=['metric', 'mean', 'std'])
    return df


def main():
    alpha = 0.191500954
    beta = 0.030798858
    extract = False
    params, max_peaks = preset_qcb_small(
        None, alpha=alpha, beta=beta, extract_chromatograms=extract)
    print(params, max_peaks)

    n_eval_episodes = 1
    fname = os.path.join('DQN', 'DQN_9_rerun_1.zip')
    rewards, eval_res = debug_run(
        fname, max_peaks, params, n_eval_episodes=n_eval_episodes, deterministic=True)
    df = eval_res_to_df(rewards, eval_res)
    print(df)

if __name__ == "__main__":
    main()