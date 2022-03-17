import numpy as np
from vimms.Evaluation import evaluate_simulated_env


def get_ppo_action_probs(model, state):
    # https://stackoverflow.com/questions/66428307/how-to-get-action-propability-in-stable-baselines-3
    obs = model.policy.obs_to_tensor(state)[0]
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().numpy()
    return probs_np


def get_ppo_best_valid_action(model, observation):
    valid_actions = observation['valid_actions']
    action_probs = get_ppo_action_probs(model, observation)
    valid_probs = action_probs * valid_actions  # set invalid actions to have 0s
    best_valid_action = np.argmax(valid_probs)
    return best_valid_action


def evaluate(env):
    res = evaluate_simulated_env(env.vimms_env)
    return res['coverage_proportion'], res['intensity_proportion']
