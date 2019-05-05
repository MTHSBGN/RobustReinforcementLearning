import gym
import numpy as np
import scipy.signal

from agents.agent import Agent
from vec_env import SubprocVecEnv
from collections import deque


class Buffer:
    def __init__(self, agent: Agent, env_name, num_envs, obs_space, act_space, num_step=64, gamma=0.99, lam=0.95):
        self.agent = agent
        self.envs = SubprocVecEnv([lambda: gym.make(env_name) for _ in range(num_envs)])
        self.num_envs = num_envs
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam

        size = (num_envs, num_step)
        self.obs_buf = np.zeros(size + obs_space.shape)
        self.act_buf = np.zeros(size + act_space.shape) if hasattr(act_space, 'shape') else np.zeros(size)
        self.rew_buf = np.zeros(size)
        self.val_buf = np.zeros(size)
        self.ret_buf = np.zeros(size)
        self.adv_buf = np.zeros(size)

        self.cum_rew = np.zeros(num_envs)
        self.rewards = deque(maxlen=100)

        self.last_obs = self.envs.reset()

    def __len__(self):
        return self.num_step

    def collect(self):
        start_idx = np.zeros(self.num_envs, dtype=np.int)
        obs = self.last_obs

        for idx in range(self.num_step):
            actions, values = self.agent.get_actions(obs, extra_returns=["value"])
            next_obs, rewards, dones, _ = self.envs.step(actions)

            self.obs_buf[:, idx] = obs
            self.act_buf[:, idx] = actions
            self.rew_buf[:, idx] = rewards
            self.val_buf[:, idx] = values
            self.cum_rew[:] += rewards

            obs = next_obs

            for i in np.argwhere(dones).reshape(-1):
                self.rewards.append(self.cum_rew[i])
                self.cum_rew[i] = 0
                self.finish_path(i, start_idx[i], idx + 1)
                start_idx[i] = idx + 1

        self.last_obs = obs

        for i in range(self.num_envs):
            self.finish_path(i, start_idx[i], self.num_step, self.val_buf[i, -1])

    def finish_path(self, idx, start, end, last_val=0):
        s = slice(start, end)
        rews = np.append(self.rew_buf[idx, s], last_val)
        vals = np.append(self.val_buf[idx, s], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

        self.ret_buf[idx, s] = discount_cumsum(rews, self.gamma)[:-1]
        self.adv_buf[idx, s] = discount_cumsum(deltas, self.gamma * self.lam)

    def get(self):
        return self.obs_buf, self.act_buf, self.ret_buf, self.adv_buf


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
