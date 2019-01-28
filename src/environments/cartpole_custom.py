import numpy as np

from gym.envs.classic_control.cartpole import CartPoleEnv


class CartPoleCustomEnv(CartPoleEnv):
    def __init__(self, gravity_bounds=None, length_bounds=None, true_value=True):
        super().__init__()
        self.gravity_bounds = gravity_bounds
        self.length_bounds = length_bounds
        self.true_value = true_value
        self.parameters = []

    def step(self, action):
        state, reward, done, info = super().step(action)
        return np.append(state, self.parameters), reward, done, info

    def reset(self, gravity=None, length=None):
        state = super().reset()

        if self.gravity_bounds:
            self.gravity = gravity if gravity else np.random.uniform(self.gravity_bounds[0], self.gravity_bounds[1])

        if self.length_bounds:
            self.length = length if length else np.random.uniform(self.length_bounds[0], self.length_bounds[1])
            self.polemass_length = self.masspole * self.length

        self.parameters = []
        self._update_parameters()

        return np.append(state, self.parameters)

    def _update_parameters(self):
        if self.gravity_bounds:
            self.parameters.append(self.gravity) if self.true_value \
                else self.parameters.append(np.random.uniform(self.gravity_bounds[0], self.gravity_bounds[1]))

        if self.length_bounds:
            self.parameters.append(self.length) if self.true_value \
                else self.parameters.append(np.random.uniform(self.length_bounds[0], self.length_bounds[1]))
