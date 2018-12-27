import gym
import numpy as np

from agents import Agent


class Evaluator:
    def __init__(self, agent: Agent, env: gym.Env):
        self.agent = agent
        self.env = env

    def evaluate(self, num_episodes=100, render=False):
        rewards = np.array([self._rollout(render) for _ in range(num_episodes)])

        stats = {
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards),
            "mean_reward": np.mean(rewards),
            "std_rewards": np.std(rewards)
        }

        print(stats)

        return np.mean(rewards) >= self.env.spec.reward_threshold and np.min(rewards) >= 150.0

    def _rollout(self, render):
        observation = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            if render:
                self.env.render()

            action = self.agent.select_actions([observation])[0]
            observation, reward, done, _ = self.env.step(action)
            total_reward += reward

        return total_reward
