import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Dict, Box, flatten_space


# modified from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/wrappers.py
# TODO: add support for discrete action space
class HistoryWrapper(gym.Wrapper):
    """
    Stack past observations and actions to give an history to the agent.
    :param env:
    :param horizon:Number of steps to keep in the history.
    """

    def __init__(self, env: gym.Env, horizon: int = 2):
        assert isinstance(env.observation_space, gym.spaces.Box)

        wrapped_obs_space = env.observation_space

        # TODO: double check, it seems wrong when we have different low and highs
        low_obs = np.repeat(wrapped_obs_space.low, horizon, axis=-1)
        high_obs = np.repeat(wrapped_obs_space.high, horizon, axis=-1)

        low = low_obs
        high = high_obs

        # Overwrite the observation space
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=wrapped_obs_space.dtype)

        super().__init__(env)

        self.horizon = horizon
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)

    def _create_obs_from_history(self):
        return self.obs_history

    def reset(self, chems=None):
        # Flush the history
        self.obs_history[...] = 0
        obs = self.env.reset(options={'chems': chems})
        self.obs_history[..., -obs.shape[-1]:] = obs
        return self._create_obs_from_history()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1]:] = obs

        return self._create_obs_from_history(), reward, done, info


# copied from https://github.com/DLR-RM/rl-baselines3-zoo/blob/feat/gymnasium-support/rl_zoo3/utils.py
def flatten_dict_observations(env: gym.Env) -> gym.Env:
    assert isinstance(env.observation_space, spaces.Dict)
    return gym.wrappers.FlattenObservation(env)


# modified from above to separate the dict space into different parts
def custom_flatten_dict_observations(env: gym.Env) -> gym.Env:
    assert isinstance(env.observation_space, spaces.Dict)
    return CustomFlattenObservation(env)


class CustomFlattenObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Observation wrapper that flattens the observation in a custom way"""

    def __init__(self, env: gym.Env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = self._flatten_space_dict(env.observation_space)

    def observation(self, observation):
        return spaces.flatten(self.env.observation_space, observation)

    def _flatten_space_dict(self, space: Dict) -> Box:
        space_list = [flatten_space(s) for s in space.spaces.values()]
        flat_space = Box(
            low=np.concatenate([s.low for s in space_list]),
            high=np.concatenate([s.high for s in space_list]),
            dtype=np.result_type(*[s.dtype for s in space_list]),
        )
        return flat_space
