import numpy as np
from vimms.Evaluation import evaluate_simulated_env, EvaluationData

from vimms_gym.common import METHOD_RANDOM, METHOD_FULLSCAN, METHOD_TOPN, METHOD_PPO, METHOD_DQN
from vimms_gym.env import DDAEnv, CoverageEnv
from vimms_gym.policy import random_policy, fullscan_policy, topN_policy, get_ppo_action_probs


class EpisodicResults():
    def __init__(self, initial_obs):
        self.env = None
        self.rewards = []
        self.observations = [initial_obs]
        self.actions = []
        self.action_probs = []
        self.infos = []
        self.num_steps = 0

    def add_step_data(self, action, action_probs, obs, reward, info):
        self.actions.append(action)
        self.action_probs.append(action_probs)
        self.observations.append(obs)
        self.rewards.append(reward)
        self.infos.append(info)
        self.num_steps += 1

    def add_environment(self, env):
        vimms_env = env.vimms_env
        self.eval_data = EvaluationData(vimms_env)
        self.eval_res = evaluate(vimms_env)
        return self.eval_res

    def get_episode_rewards(self):
        return np.sum(self.rewards)


def evaluate(env):
    # env can be either a DDAEnv or a ViMMS' Environment object
    try:
        vimms_env = env.vimms_env
    except AttributeError:
        vimms_env = env

    vimms_env_res = evaluate_simulated_env(vimms_env)
    count_fragmented = np.count_nonzero(vimms_env_res['times_fragmented'])
    count_ms1 = len(vimms_env.controller.scans[1])
    count_ms2 = len(vimms_env.controller.scans[2])
    ms1_ms2_ratio = float(count_ms1) / count_ms2
    efficiency = float(count_fragmented) / count_ms2

    eval_res = {
        'coverage_prop': '%.3f' % vimms_env_res['coverage_proportion'][0],
        'intensity_prop': '%.3f' % vimms_env_res['intensity_proportion'][0],
        'ms1/ms2 ratio': '%.3f' % ms1_ms2_ratio,
        'efficiency': '%.3f' % efficiency,
    }
    return eval_res


def run_method(env_name, env_params, max_peaks, chem_list, method, out_dir,
               N=10, min_ms1_intensity=5000, model=None,
               print_eval=False, print_reward=False, mzml_prefix=None):
    if method in [METHOD_DQN, METHOD_PPO]:
        assert model is not None

    # to store all results across all loop of chem_list
    all_episodic_results = []

    for i in range(len(chem_list)):
        chems = chem_list[i]
        if print_reward:
            print(f'\nEpisode {i} ({len(chems)} chemicals)')

        if env_name == 'DDAEnv':
            env = DDAEnv(max_peaks, env_params)
        elif env_name == 'CoverageEnv':
            env = CoverageEnv(max_peaks, env_params)

        obs = env.reset(chems=chems)
        done = False

        # lists to store episodic results
        er = EpisodicResults(obs)
        while not done:  # repeat until episode is done

            # select an action depending on the observation and method
            action, action_probs = pick_action(
                method, obs, model, env.features, N, min_ms1_intensity)

            # make one step through the simulation
            obs, reward, done, info = env.step(action)

            # store new episodic information
            if obs is not None:
                er.add_step_data(action, action_probs, obs, reward, info)

            if print_reward and er.num_steps % 500 == 0:
                print('steps\t', er.num_steps, '\ttotal rewards\t', er.get_episode_rewards())

            # if episode is finished, break
            if done:
                break

        if print_reward:
            print(
                f'Finished after {er.num_steps} timesteps with '
                f'total reward {er.get_episode_rewards()}')

        # save mzML and other info useful for evaluation of the ViMMS environment
        if mzml_prefix is None:
            out_file = '%s_%d.mzML' % (method, i)
        else:
            out_file = '%s_%s_%d.mzML' % (mzml_prefix, method, i)
        env.write_mzML(out_dir, out_file)

        # environment will be evaluated here
        eval_res = er.add_environment(env)
        if print_eval:
            print(eval_res)
        all_episodic_results.append(er)

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
        # action = best_ppo_policy(obs, model)
        action_probs = get_ppo_action_probs(model, obs)
    elif method == METHOD_DQN:
        action, _states = model.predict(obs, deterministic=True)

    return action, action_probs
