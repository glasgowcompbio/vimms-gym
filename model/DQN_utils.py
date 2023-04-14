import random
import socket

import numpy as np
import torch


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
