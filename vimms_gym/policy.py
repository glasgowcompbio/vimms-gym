import numpy as np

from vimms_gym.evaluation import get_ppo_action_probs
from vimms_gym.features import obs_to_dfs


def fullscan_policy(obs):
    """
    A policy function that generates only fullscan data.
    """
    valid_actions = obs['valid_actions']
    ms1_action = len(valid_actions)-1 # last index is always the action for MS1 scan
    return ms1_action


def random_policy(obs):
    """
    A policy function that selects a random valid move from the observation.
    """

    # randomly choose one valid action based on the observation
    # if there are some ions to fragment, we could either fragment one or perform an MS1 scan
    # if nothing to fragment, then perform an MS1 scan
    valid_actions = obs['valid_actions']
    nnz = np.nonzero(valid_actions)[0]  # valid_actions are the non-zeros
    nnz_idx = np.random.choice(len(nnz))
    action = nnz[nnz_idx]
    return action


def topN_policy(obs, N, min_ms1_intensity):
    """
    A policy function that performs TopN selection from the observation.
    """
    # turn observation dictionary to dataframe
    scan_df, count_df = obs_to_dfs(obs)

    # set an indicator column for min intensity check
    scan_df['above_min_intensity'] = 1
    scan_df.loc[scan_df['intensities'] < np.log(min_ms1_intensity), 'above_min_intensity'] = 0

    # check whether N ions have been fragmented
    fragmented_count = count_df.loc['fragmented_count'].values[0]
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
        idx = filtered['intensities'].idxmax()
        return idx


def best_ppo_policy(obs, model):
    valid_actions = obs['valid_actions']
    action_probs = get_ppo_action_probs(model, obs)
    valid_probs = action_probs * valid_actions  # set invalid actions to 0 probabilities
    best_valid_action = np.argmax(valid_probs)
    return best_valid_action
