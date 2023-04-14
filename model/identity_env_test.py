# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from identity_env import IdentityEnv, IdentityEnvDiscrete


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
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
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


def make_env(env_id, seed):
    def thunk():

        # create custom environment here if needed
        if env_id == 'identity':
            env = IdentityEnv(dim=10)

        elif env_id == 'identity_masked':
            env = IdentityEnvDiscrete(dim=10)

        else:  # registered environment
            env = gym.make(env_id)

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

        in_features = int(np.array(env.single_observation_space.shape).prod())
        self.network = nn.Sequential(
            nn.Linear(in_features, 4),
            nn.ReLU(),
            nn.Linear(4, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def evaluate(
        model_path: str,
        make_env: Callable,
        env_id: str,
        eval_episodes: int,
        Model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        epsilon: float = 0.05,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, infos = envs.step(actions)
        for info in infos:
            if "episode" in info.keys():
                print(
                    f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


def main(args):
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i) for i in range(NUM_ENVS)])
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

        # Implement action masking. All the codes here assumes that only a single environment
        # is used, as that's the limitation of the replay buffer anyway.
        env = envs.envs[0].env
        try:
            masks = env.action_masks()
            if random.random() < epsilon:
                # masked epsilon move
                valid_actions = np.argwhere(masks == 1).flatten()
                actions = np.array([np.random.choice(valid_actions)])
            else:
                q_values = q_network(torch.Tensor(obs).to(device))
                # masked greedy move
                q_values_np = q_values.detach().cpu().numpy()
                min_value = 1E-10
                masked_q_values_np = np.where(masks, q_values_np, min_value)
                actions = np.array([np.argmax(masked_q_values_np)])

        except AttributeError: # no masking
            if random.random() < epsilon:
                actions = np.array([env.action_space.sample()]) # epsilon move
            else:
                q_values = q_network(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy() # greedy move

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
                    x = data.next_observations.float()
                    target_max, _ = target_network(x).max(dim=1)
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
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()

    print(f"mean_episodic_return={np.mean(episodic_return)}")


if __name__ == "__main__":
    args = parse_args()

    # Set the desired arguments
    args.exp_name = "dqn_test"
    args.seed = 42
    # args.env_id = "CartPole-v1"
    # args.env_id = 'identity'
    args.env_id = 'identity_masked'
    args.total_timesteps = 500000
    args.learning_rate = 2.5e-4
    args.buffer_size = 10000
    args.gamma = 0.99
    args.tau = 1.
    args.target_network_frequency = 500
    args.batch_size = 128
    args.start_e = 1
    args.end_e = 0.05
    args.exploration_fraction = 0.5
    args.learning_starts = 10000
    args.train_frequency = 10

    # Call the main training loop
    main(args)