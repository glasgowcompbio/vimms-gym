from typing import Callable

import gymnasium as gym
import pandas as pd
import torch
from vimms.Common import load_obj
from vimms.Evaluation import EvaluationData

from model.DQN_utils import masked_epsilon_greedy
from vimms_gym.evaluation import get_task_params
from vimms_gym.common import evaluate


def evaluate_model(
        model_path: str,
        make_env: Callable,
        env_id: str,
        task: str,
        eval_episodes: int,
        Model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
):
    # env setup
    max_peaks, params, chem_path = get_task_params(task)
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, max_peaks, params)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # load evaluation dataset
    chem_list = load_obj(chem_path)
    episodic_returns = []
    evaluation_results = []
    for i in range(len(chem_list)):
        if i >= eval_episodes:
            break

        chems = chem_list[i]

        obs, info = envs.reset(options={'chems': chems})
        done = False
        eval_data = None
        while not done:
            env = envs.envs[0].env
            actions = masked_epsilon_greedy(device, max_peaks, None, obs, model,
                                            deterministic=True)

            next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)
            dones = terminateds

            if 'final_info' in infos:
                final_infos = infos['final_info']
                assert len(final_infos) == 1
                info = final_infos[0]
                episodic_return = info['episode']['r'][0]
                episodic_length = info['episode']['l'][0]

                eval_res = evaluate(eval_data)
                eval_res['invalid_action_count'] = env.invalid_action_count
                eval_res['total_rewards'] = episodic_return
                eval_res['episodic_length'] = episodic_length
                eval_res['num_ms1_scans'] = len(eval_data.controller.scans[1])
                eval_res['num_ms2_scans'] = len(eval_data.controller.scans[2])
                evaluation_results.append(eval_res)

                print(
                    f'Episode {i} ({len(chems)} chemicals) return {episodic_return} length {episodic_length}')
                print(eval_res)

                episodic_returns += [episodic_return]

            obs = next_obs
            done = dones[0]

            # store previous results for evaluation before 'done'
            # this needs to be here, because VecEnv is automatically reset when done
            vimms_env = env.env.vimms_env
            eval_data = EvaluationData(vimms_env)

    df = pd.DataFrame(evaluation_results)
    df = df.astype(float)
    pd.set_option('display.max_columns', None)
    df_summary = df.describe(include='all')
    print(df_summary)
    # df_summary.to_csv('df_summary.tsv', sep='\t')
    # df.to_csv('df.tsv', sep='\t')

    return episodic_returns, df, df_summary
