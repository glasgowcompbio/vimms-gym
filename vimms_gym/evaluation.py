import numpy as np
from vimms.Evaluation import evaluate_simulated_env

from vimms_gym.policy import random_policy, fullscan_policy, topN_policy, best_ppo_policy


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


def run_method(env, chem_list, method, out_dir, N=10, min_ms1_intensity=5000, model=None,
               print_eval=False, print_reward=False, mzml_prefix=None):
    if method in ['DQN', 'PPO']:
        assert model is not None

    eval_results = []
    for i in range(len(chem_list)):
        chems = chem_list[i]
        observation = env.reset(chems=chems)
        done = False
        num_steps = 0
        episode_reward = 0

        while not done:

            if method == 'random':
                action = random_policy(observation)
            elif method == 'fullscan':
                action = fullscan_policy(observation)
            elif method == 'TopN':
                action = topN_policy(observation, env.features, N, min_ms1_intensity)
            elif method == 'DQN':
                action, _states = model.predict(observation, deterministic=True)
            elif method == 'PPO':
                # action = best_ppo_policy(observation, model)
                action, _states = model.predict(observation, deterministic=True)

            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if print_reward and num_steps % 500 == 0:
                print(num_steps, episode_reward)
            num_steps += 1

            if done:
                print(
                    f'Episode {i} finished after {num_steps} timesteps with reward {episode_reward}')
                if mzml_prefix is None:
                    out_file = '%s_%d.mzML' % (method, i)
                else:
                    out_file = '%s_%s_%d.mzML' % (mzml_prefix, method, i)

                env.write_mzML(out_dir, out_file)
                eval_res = evaluate(env)
                eval_results.append(eval_res)
                if print_eval:
                    print(eval_res)
                break
    return eval_results
