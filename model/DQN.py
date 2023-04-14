import argparse
import os
import random
import time
from datetime import datetime
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from model.DQN_utils import set_torch_threads, masked_epsilon_greedy, get_action_masks_from_obs
from model.QNetwork import QNETWORK_CNN, get_QNetwork
from model.evaluation import evaluate_model
from vimms_gym.common import METHOD_DQN
from vimms_gym.env import DDAEnv
from vimms_gym.experiments import preset_qcb_medium
from vimms_gym.wrappers import custom_flatten_dict_observations


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True,
                        nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?",
                        const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?",
                        const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="DDAEnv",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True,
                        nargs="?", const=True,
                        help="whether to save model into the `runs/{run_name}` folder")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
                        help="the id of the environment")
    parser.add_argument("--qnetwork", type=str, default=QNETWORK_CNN,
                        help="the type of Q-Network to use")
    parser.add_argument("--total-timesteps", type=int, default=int(1E5),
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=0.0005,
                        help="the learning rate of the optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.0001,
                        help="the weight decay of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(5E4),
                        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
                        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=8000,
                        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.02,
                        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=50000,
                        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
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


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


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
            settings=wandb.Settings(code_dir=os.path.join('..', 'vimms_gym'))
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

    q_network = get_QNetwork(args.qnetwork, envs, device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate,
                           weight_decay=args.weight_decay)
    target_network = get_QNetwork(args.qnetwork, envs, device)
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
            Model=get_QNetwork(args.qnetwork, None, None, initialise=False),
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

    envs.close()
    writer.close()

    print(f"mean_episodic_return={np.mean(episodic_return)}")


if __name__ == "__main__":
    args = parse_args()

    # Call the main training loop
    set_torch_threads()
    main(args)
