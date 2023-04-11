# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import random
import socket
import time
from datetime import datetime
from distutils.util import strtobool
from typing import Callable

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from vimms.Common import load_obj
from vimms.Evaluation import EvaluationData

from vimms_gym.common import METHOD_DQN, evaluate
from vimms_gym.env import DDAEnv
from vimms_gym.experiments import preset_qcb_medium
from vimms_gym.wrappers import custom_flatten_dict_observations


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True,
                        nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?",
                        const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?",
                        const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="DDAEnv",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False,
                        nargs="?", const=True,
                        help="whether to save model into the `runs/{run_name}` folder")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
                        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
                        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
                        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
                        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
                        help="the frequency of training")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, max_peaks, params):
    def thunk():
        env = DDAEnv(max_peaks, params)
        check_env(env)
        env = custom_flatten_dict_observations(env)

        env.reset(seed=seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.n_hidden = [256, 256]
        self.roi_network_out = 64
        self.n_total_features = 427
        self.n_roi = 30
        self.roi_length = 10
        self.n_roi_features = self.n_roi * self.roi_length  # 30 rois, each is length 10, so total is 300 features
        self.n_other_features = self.n_total_features - self.n_roi_features  # the remaining, which is 247 features

        # configuration 1

        self.roi_network = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(start_dim=1),
            nn.Linear(64, self.roi_network_out),
            nn.ReLU(),
        )

        input_size = (self.roi_network_out * self.n_roi) + self.n_hidden[1]
        output_size = env.single_action_space.n
        self.output_layer = nn.Linear(input_size, output_size)

        self.other_network = nn.Sequential(
            nn.Linear(self.n_other_features, self.n_hidden[0]),
            nn.ReLU(),
            nn.Linear(self.n_hidden[0], self.n_hidden[0]),
            nn.ReLU(),
            nn.Linear(self.n_hidden[0], self.n_hidden[1]),
            nn.ReLU(),
        )

    def forward(self, x):
        # get dense network prediction for other features
        other_inputs = x[:, self.n_roi_features:]
        other_output = self.other_network(other_inputs)

        # transform ROI input to the right shape: (self.n_roi, self.roi_length)
        roi_inputs = x[:, 0:self.n_roi_features]
        roi_img_inputs = roi_inputs.view(-1, self.n_roi, self.roi_length)

        # Reshape the tensor to (batch_size * num_roi, 1, num_features)
        roi_img_inputs_reshaped = roi_img_inputs.reshape(-1, 1, self.roi_length)

        # Process each ROI separately
        roi_output = self.roi_network(roi_img_inputs_reshaped)

        # Reshape the output
        # flatten the output for all ROIs
        roi_output = roi_output.view(other_output.shape[0], -1)

        # average across ROIs -- doesnt' work well
        # roi_output = roi_output.view(-1, self.n_roi, self.n_hidden[1])
        # roi_output = torch.mean(roi_output, dim=1)

        # Concatenate the outputs of the two networks
        combined_output = torch.cat((roi_output, other_output), dim=-1)

        # Generate Q-value predictions
        q_values = self.output_layer(combined_output)
        return q_values


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def evaluate_model(
        model_path: str,
        make_env: Callable,
        env_id: str,
        eval_episodes: int,
        Model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        epsilon: float = 0.05,
        intensity_threshold: float = 5000
):
    params, max_peaks = preset_qcb_medium(METHOD_DQN, alpha=0.00, beta=0.00,
                                          extract_chromatograms=True)
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, max_peaks, params)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # load evaluation dataset
    chem_path = os.path.join('..', 'notebooks', 'QCB_resimulated_medium', 'QCB_chems_medium.p')
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
            actions = masked_epsilon_greedy(device, max_peaks, epsilon, obs, model,
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
    df_summary.to_csv('df_summary.tsv', sep='\t')
    df.to_csv('df.tsv', sep='\t')

    return episodic_returns, df, df_summary


def main(args):
    current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    run_name = f"{args.env_id}__{args.exp_name}__{current_time}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    NUM_ENVS = 1
    params, max_peaks = preset_qcb_medium(METHOD_DQN, alpha=0.00, beta=0.00,
                                          extract_chromatograms=True)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, max_peaks, params) for i in range(NUM_ENVS)])
    assert isinstance(envs.single_action_space,
                      gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # TODO: ReplayBuffer only supports a single environment for now
    assert NUM_ENVS == 1
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    total_returns = []
    obs, _ = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e,
                                  args.exploration_fraction * args.total_timesteps, global_step)

        # Normal epsilon-greedy move with no action masking
        # actions = epsilon_greedy(device, env, epsilon, obs, q_network)

        # Implement action masking. All the codes here assumes that only a single environment
        # is used, as that's the limitation of the replay buffer anyway.
        actions = masked_epsilon_greedy(device, max_peaks, epsilon, obs, q_network)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)
        dones = terminateds

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if 'final_info' in infos:
            final_infos = infos['final_info']
            for info in final_infos:
                idx = 0
                episodic_return = info['episode']['r'][0]
                episodic_length = info['episode']['l'][0]
                total_returns.append(episodic_return)
                print(f"global_step={global_step}, episodic_return={episodic_return}, "
                      f"episodic_length={episodic_length}")
                writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                writer.add_scalar("charts/episodic_length", episodic_length, global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():

                    # shape is (batch_size, num_features)
                    x = data.next_observations.float()
                    action_masks = get_action_masks_from_obs(x, max_peaks)

                    # Apply the mask to the target network's output
                    target_q_values = target_network(x)
                    min_value = float('-inf')
                    masked_target_q_values = torch.where(action_masks, target_q_values,
                                                         torch.tensor(min_value).to(x.device))
                    target_max, _ = masked_target_q_values.max(dim=1)

                    td_target = data.rewards.flatten() + args.gamma * target_max * (
                            1 - data.dones.flatten())

                x = data.observations.float()
                old_val = q_network(x).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)),
                                      global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(),
                                                                 q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (
                                1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

        EVAL_EPISODES = 30
        episodic_returns, df, df_summary = evaluate_model(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=EVAL_EPISODES,
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )

        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episode/reward", episodic_return, idx)
            for col in df.columns:
                val = df.loc[idx, col]
                writer.add_scalar("eval/episode/%s" % col, val, idx)

        mean_cols = df_summary.loc[['mean']]
        for column in mean_cols.columns:
            mean_value = mean_cols.at['mean', column]
            wandb.log({f"eval/means/{column}": mean_value})

        # mean_values = mean_cols.to_dict()
        # mean_dict = {key: value['mean'] for key, value in mean_values.items()}
        # wandb.log({"mean_values": wandb.Table(data=[mean_dict.values()], columns=["Value"],
        #                                       index=mean_dict.keys())})
        # table_data = [[key, value] for key, value in mean_dict.items()]
        # wandb.log({"mean_values": wandb.Table(data=table_data, columns=["Key", "Value"])})

    envs.close()
    writer.close()

    print(f"mean_episodic_return={np.mean(episodic_return)}")


def get_action_masks_from_obs(obs, max_peaks):
    # extract the last max_peaks+1 columns in obs and turn them into a boolean mask
    n_mask = max_peaks + 1

    if isinstance(obs, torch.Tensor):
        action_masks = obs[:, -n_mask:].bool()
    elif isinstance(obs, np.ndarray):
        action_masks = obs[:, -n_mask:].astype(bool)
    else:
        raise TypeError("Unsupported input type {}".format(type(obs)))

    return action_masks


def masked_epsilon_greedy(device, max_peaks, epsilon, obs, q_network, deterministic=False):
    masks = get_action_masks_from_obs(obs, max_peaks)
    if deterministic:  # exploitation only
        actions = masked_greedy(device, masks, obs, q_network)

    else:  # exploration and exploitation
        if random.random() < epsilon:
            actions = masked_epsilon(masks)
        else:
            actions = masked_greedy(device, masks, obs, q_network)
    return actions


def masked_greedy(device, masks, obs, q_network):
    q_values = q_network(torch.Tensor(obs).to(device))
    # masked greedy move
    q_values_np = q_values.detach().cpu().numpy()
    min_value = float('-inf')
    masked_q_values_np = np.where(masks, q_values_np, min_value)
    actions = np.array([np.argmax(masked_q_values_np)])
    return actions


def masked_epsilon(masks):
    # masked epsilon move
    valid_actions = np.argwhere(masks == 1).flatten()
    actions = np.array([np.random.choice(valid_actions)])
    return actions


def epsilon_greedy(device, env, epsilon, obs, q_network):
    if random.random() < epsilon:
        actions = np.array([env.action_space.sample()])  # epsilon move
    else:
        q_values = q_network(torch.Tensor(obs).to(device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()  # greedy move
    return actions


def set_torch_threads():
    torch_threads = 1  # Set pytorch num threads to 1 for faster training
    if socket.gethostname() == 'cauchy':  # except on cauchy where we have no gpu, only cpu
        torch_threads = 40
    torch.set_num_threads(torch_threads)


if __name__ == "__main__":
    args = parse_args()

    # Set the desired arguments
    args.exp_name = "DQN"
    args.seed = 42

    args.env_id = 'DDAEnv'
    args.total_timesteps = int(1E5)
    args.learning_rate = 0.0005
    args.buffer_size = int(5E4)
    args.gamma = 0.99
    args.tau = 1.
    args.target_network_frequency = 8000
    args.batch_size = 64
    args.start_e = 1
    args.end_e = 0.02
    args.exploration_fraction = 0.1
    args.learning_starts = 50000
    args.train_frequency = 4
    args.save_model = True
    args.track = True

    # Call the main training loop
    set_torch_threads()
    main(args)
