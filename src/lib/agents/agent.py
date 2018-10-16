from gym.spaces import Box, Discrete

from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, action_space, observation_space):
        if isinstance(action_space, Discrete):
            actions = list(range(action_space.n))

        self.actions = actions

        if isinstance(observation_space, Box):
            num_obs = len(observation_space.low)

        self.observation_dim = num_obs

    @abstractmethod
    def select_action(self, observation):
        pass

    @abstractmethod
    def improve(self, rewards):
        pass
