import argparse
import pprint
from datetime import datetime

import gymnasium as gym
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from vimms.Common import load_obj
from vimms.Evaluation import EvaluationData

import wandb
from model.DQN_utils import make_env, masked_epsilon_greedy, set_torch_threads
from model.QNetwork import QNETWORK_CNN
from train_DQN import training_loop
from vimms_gym.evaluation import get_task_params
from vimms_gym.common import evaluate
from vimms_gym.experiments import ENV_QCB_MEDIUM_EXTRACTED, \
    ENV_QCB_LARGE_EXTRACTED, ENV_QCB_SMALL_EXTRACTED, ENV_QCB_SMALL_GAUSSIAN, \
    ENV_QCB_MEDIUM_GAUSSIAN, ENV_QCB_LARGE_GAUSSIAN


# Define objective/training function
def objective(model, env_id, device, writer, eval_episodes, task):
    max_peaks, params, chem_path = get_task_params(task)
    model.eval()
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, max_peaks, params)])

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
    df_summary = df.describe(include='all')

    for idx, episodic_return in enumerate(episodic_returns):
        writer.add_scalar("eval/episode/reward", episodic_return, idx)
        for col in df.columns:
            val = df.loc[idx, col]
            writer.add_scalar("eval/episode/%s" % col, val, idx)

    mean_cols = df_summary.loc[['mean']]
    for column in mean_cols.columns:
        mean_value = mean_cols.at['mean', column]
        wandb.log({f"eval/means/{column}": mean_value})

    mean_f1 = mean_cols.at['mean', 'f1']
    print(f"mean_f1={mean_f1}")
    return mean_f1


def parse_args():
    parser = argparse.ArgumentParser(description='Script Parameters')
    parser.add_argument('--total_timesteps', type=lambda x: int(float(x)), default=1E6,
                        help='Total timesteps for training')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training')
    parser.add_argument('--qnetwork', type=str, default=QNETWORK_CNN,
                        help='Type of Q Network. Can be "CNN", "LSTM", "DENSE" or "DENSE_FLAT"')
    parser.add_argument('--eval_episodes', type=int, default=30,
                        help='No. of evaluation episodes, should be between 0 to 30')
    parser.add_argument('--sweep_count', type=int, default=30,
                        help='No. of sweep trials')
    parser.add_argument('--sweep_method', type=str, default='bayes',
                        help='Choice of parameter sweep method. Can be "random" or "bayes"')

    task_choices = [
        ENV_QCB_SMALL_GAUSSIAN, ENV_QCB_MEDIUM_GAUSSIAN, ENV_QCB_LARGE_GAUSSIAN,
        ENV_QCB_SMALL_EXTRACTED, ENV_QCB_MEDIUM_EXTRACTED, ENV_QCB_LARGE_EXTRACTED,
    ]
    parser.add_argument("--task", type=str, default=ENV_QCB_MEDIUM_EXTRACTED,
                        choices=task_choices, help="type of tasks")

    args = parser.parse_args()
    return args


def sweep():
    wandb.init()

    # fixed values
    seed = 42
    torch_deterministic = True
    num_envs = 1
    env_type = 'sync'
    env_id = 'DDAEnv'
    exp_name = 'DQN'
    start_e = 1.0
    tau = 1.0

    # parameterise using argparse
    total_timesteps = args.total_timesteps
    batch_size = args.batch_size
    qnetwork = args.qnetwork
    eval_episodes = args.eval_episodes
    task = args.task

    # wandb sweep values
    learning_rate = wandb.config.learning_rate
    weight_decay = wandb.config.weight_decay
    buffer_size = wandb.config.buffer_size
    end_e = wandb.config.end_e
    exploration_fraction = wandb.config.exploration_fraction
    learning_starts = wandb.config.learning_starts
    train_frequency = wandb.config.train_frequency
    gamma = wandb.config.gamma
    target_network_frequency = wandb.config.target_network_frequency

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{env_id}_{exp_name}_{current_time}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    model = training_loop(
        seed, torch_deterministic, num_envs, env_type, env_id,
        qnetwork, learning_rate, weight_decay, buffer_size, total_timesteps,
        start_e, end_e, exploration_fraction, learning_starts, train_frequency,
        batch_size, gamma, target_network_frequency, tau, task,
        device, writer
    )

    score = objective(model, env_id, device, writer, eval_episodes, task)
    writer.close()

    wandb.log({'run:createdAt': current_time})
    wandb.log({'summary:f1': score})


# Define the search space
sweep_configuration = {
    'metric':
        {
            'goal': 'maximize',
            'name': 'f1'
        },
    'parameters':
        {
            'learning_rate': {'max': 1.0, 'min': 1e-5, 'distribution': 'log_uniform_values'},
            'weight_decay': {'values': [0.0, 0.0001, 0.0010, 0.01, 0.1]},
            'buffer_size': {'values': [int(5E4), int(1E5), int(2E5), int(5E5)]},
            'end_e': {'max': 0.2, 'min': 0.0},
            'exploration_fraction': {'max': 0.5, 'min': 0.0},
            'learning_starts': {'values': [int(5E4), int(1E5), int(2E5), int(5E5)]},
            'train_frequency': {'values': [1, 4, 8, 16, 128, 256, 1000]},
            'gamma': {'values': [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]},
            'target_network_frequency': {'values': [1, 1000, 5000, 8000, 10000, 15000, 20000]}
        }
}

if __name__ == "__main__":
    args = parse_args()
    print('args')
    pprint.pprint(vars(args))

    set_torch_threads()
    wandb.login()

    # start a new sweep
    sweep_configuration['method'] = args.sweep_method
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='DDAEnv')
    wandb.agent(sweep_id, function=sweep, count=args.sweep_count)

    # continue existing sweep
    # sweep_id = 't4op68lf'
    # wandb.agent(sweep_id, project='DDAEnv', function=sweep, count=args.sweep_count)