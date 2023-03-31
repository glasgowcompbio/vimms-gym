from typing import List
from typing import Optional, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymStepReturn


class IdentityEnv(gym.Env):
    def __init__(self, dim: Optional[int] = None, ep_length: int = 100):
        """
        Identity environment for testing purposes
        :param dim: the size of the action and observation dimension you want
            to learn.
        :param ep_length: the length of each episode in timesteps
        """
        if dim is None:
            dim = 1
        obs_space = spaces.MultiDiscrete([dim])

        self.action_space = spaces.Discrete(dim)
        self.observation_space = obs_space
        self.ep_length = ep_length
        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.
        self.reset()

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.current_step = 0
        self.num_resets += 1
        self._choose_next_state()
        info = {}
        return self.state, info

    def step(self, action: Union[int, np.ndarray]) -> GymStepReturn:
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        terminated = self.current_step >= self.ep_length
        truncated = False
        info = {}
        return self.state, reward, terminated, truncated, info

    def _choose_next_state(self) -> None:
        self.state = self.observation_space.sample()

    def _get_reward(self, action: Union[int, np.ndarray]) -> float:
        return 1.0 if np.all(self.state == action) else 0.0

    def render(self, mode: str = "human") -> None:
        pass


class IdentityEnvDiscrete(IdentityEnv):
    def __init__(self, dim: int = 1, ep_length: int = 100):
        """
        Identity environment for testing purposes

        :param dim: the size of the dimensions you want to learn
        :param ep_length: the length of each episode in timesteps
        """
        self.useless_property = 1
        super().__init__(ep_length=ep_length, dim=dim)

    def action_masks(self) -> List[int]:
        return np.array([i == self.state for i in range(self.action_space.n)]).flatten()
