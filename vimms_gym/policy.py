import numpy as np
import torch as th

from vimms_gym.features import obs_to_dfs


def fullscan_policy(obs):
    """
    A policy function that generates only fullscan data.
    """
    valid_actions = obs['valid_actions']
    ms1_action = len(valid_actions) - 1  # last index is always the action for MS1 scan
    return ms1_action


def random_policy(obs, valid=False):
    """
    A policy function that selects a random valid move from the observation.
    """

    if not valid: # randomly choose one valid action, even if it's invalid
        action = np.random.choice(len(obs['valid_actions']))

    else: # harder random baseline -- only chooses among valid actions
        valid_actions = obs['valid_actions']
        valid_indices = np.where(valid_actions)[0]
        action = np.random.choice(valid_indices)

    return action


def topN_policy(obs, features, N, min_ms1_intensity):
    """
    A policy function that performs TopN selection from the observation.
    """
    # turn observation dictionary to dataframe
    # print(obs)
    scan_df, count_df = obs_to_dfs(obs, features)

    # set an indicator column for min intensity check
    scan_df['above_min_intensity'] = 1
    scan_df.loc[scan_df['log_intensities'] < np.log(min_ms1_intensity), 'above_min_intensity'] = 0

    # check whether N ions have been fragmented
    scaled_fragmented_count = count_df.loc['num_fragmented'].values[0]
    fragmented_count = np.round(scaled_fragmented_count * len(features))
    if fragmented_count >= N:
        return fullscan_policy(obs)  # if yes, do an MS1 scan

    # TopN selection: find unfragmented, unexcluded peaks above min_intensity
    filtered = scan_df.query(
        'fragmented == 0 & excluded == 0 & above_min_intensity == 1')

    # nothing to fragment, do an MS1 scan
    if len(filtered) == 0:
        return fullscan_policy(obs)

    # select the most intense peak to fragment
    else:
        idx = filtered['_roi_intensities_1'].idxmax()

        # some checks on the selected feature to fragment
        f = features[idx]
        assert f.intensity > min_ms1_intensity # check above min intensity
        assert not f.excluded # check not currently excluded
        assert not f.fragmented # check not fragmented before
        return idx


def best_ppo_policy(obs, model):
    valid_actions = obs['valid_actions']
    action_probs = get_ppo_action_probs(model, obs)
    valid_probs = action_probs * valid_actions  # set invalid actions to 0 probabilities
    best_valid_action = np.argmax(valid_probs)
    return best_valid_action


def get_ppo_action_probs(model, state):
    # https://stackoverflow.com/questions/66428307/how-to-get-action-propability-in-stable-baselines-3
    obs = model.policy.obs_to_tensor(state)[0]
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().cpu().numpy()
    return probs_np


def get_recurrent_ppo_action_probs(model, state, lstm_states, episode_starts):
    obs = model.policy.obs_to_tensor(state)[0]
    dis = model.policy.get_distribution(obs, lstm_states, episode_starts)
    probs = dis.distribution.probs
    probs_np = probs.detach().cpu().numpy()
    return probs_np


def get_ppo_best_valid_action(model, observation):
    valid_actions = observation['valid_actions']
    action_probs = get_ppo_action_probs(model, observation)
    valid_probs = action_probs * valid_actions  # set invalid actions to have 0s
    best_valid_action = np.argmax(valid_probs)
    return best_valid_action


def get_dqn_q_values(model, observation):
    with th.no_grad():
        obs_tensor, _ = model.q_net.obs_to_tensor(observation)
        q_values = model.q_net(obs_tensor)
        return q_values