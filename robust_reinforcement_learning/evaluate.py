import gym
import numpy as np

from robust_reinforcement_learning.agents.ppo_agent import PPOAgent


def evaluate(env_name, agent, n=100):
    env = gym.make(env_name)
    rewards = []
    for _ in range(n):
        obs = env.reset()
        done = False
        cum_reward = 0
        while not done:
            _, action, _ = agent([obs], None)
            obs, reward, done, _ = env.step(action[0])
            cum_reward += reward
        rewards.append(cum_reward)
    return np.mean(rewards)


if __name__ == "__main__":
    env_name = "BipedalWalker-v2"
    env = gym.make(env_name)
    agent = PPOAgent(env.observation_space, env.action_space)
    agent.load()
    evaluate(env_name)
