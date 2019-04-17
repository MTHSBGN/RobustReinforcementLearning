import gym
import numpy as np

from agents.agent import Agent
from vec_env import SubprocVecEnv


class Evaluator:
    def __init__(self, env_name, num_envs):
        self.env = SubprocVecEnv([lambda: gym.make(env_name) for _ in range(num_envs)])
        self.num_envs = num_envs

    def evaluate(self, agent: Agent, num_rewards=100):
        rewards = []
        cum_reward = np.zeros(self.num_envs)
        obs = self.env.reset()

        while len(rewards) < num_rewards:
            actions = agent.get_actions(obs)
            obs, reward, done, _ = self.env.step(actions)
            cum_reward += reward

            for i in np.argwhere(done).reshape(-1):
                rewards.append(cum_reward[i])
                cum_reward[i] = 0

        return np.mean(rewards)
