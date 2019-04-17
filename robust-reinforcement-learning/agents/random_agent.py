import gym
import numpy as np

from agents.agent import Agent


class RandomAgent(Agent):
    def get_actions(self, observations):
        assert len(observations.shape) > 1

        if type(self.act_space) is gym.spaces.Discrete:
            return np.random.randint(0, self.act_space.n, len(observations))

        return np.random.uniform(self.act_space.low, self.act_space.high, (len(observations),) + self.act_space.shape)

    def update(self, data):
        pass
