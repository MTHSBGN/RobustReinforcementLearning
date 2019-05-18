import numpy as np


class Agent:
    def __init__(self, env, model=None):
        self.action_space = env.action_space
        self.model = model

    def select_action(self, obs):
        if not self.model:
            return np.random.uniform(
                self.action_space.low,
                self.action_space.hivigh,
                (len(obs), self.action_space.shape[0])
            )

        _, act, value = self.model(obs, None)
        return act, value

    def train(self, data):
        if self.model:
            return self.model.update(data)
