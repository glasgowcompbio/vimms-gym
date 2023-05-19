import random
import socket

import gymnasium as gym
import torch
from gymnasium.utils.env_checker import check_env
import numpy as np

from vimms_gym.env import DDAEnv
from vimms_gym.wrappers import custom_flatten_dict_observations


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
    masked_q_values_np = q_values_np.copy()
    masked_q_values_np[~masks] = min_value
    actions = np.argmax(masked_q_values_np, axis=1).reshape(-1, 1)  # reshape for Nx1

    return actions


def masked_epsilon(masks):
    # masked epsilon move
    actions = select_random_true_positions(masks)
    return actions


def select_random_true_positions(mask):
    selected_positions = np.zeros(mask.shape[0], dtype=int)

    for idx, row in enumerate(mask):
        true_positions = np.where(row)[0]  # Get the indices where value is True
        selected_positions[idx] = np.random.choice(true_positions)  # Randomly select one of the True positions

    return selected_positions.reshape(-1, 1)  # reshape for mask.shape[0]x1

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
