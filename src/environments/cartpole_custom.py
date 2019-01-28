import numpy as np

from gym import spaces
from gym.envs.classic_control.cartpole import CartPoleEnv


class CartPoleCustomEnv(CartPoleEnv):
    def __init__(self, gravity=None, length=None, true_value=True):
        super().__init__()

        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        low = -high

        for bounds in [gravity, length]:
            if bounds:
                low = np.append(low, bounds[0])
                high = np.append(high, bounds[1])

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.gravity_bounds = gravity
        self.length_bounds = length
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
