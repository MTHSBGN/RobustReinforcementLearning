import gym
import numpy as np

from lib.agents import Agent


class Experiment:
    def __init__(self, agent_name, env_name, num_episodes=1000, batch_episode=2, summary=False, render=False):
        self.simulator = gym.make(env_name)
        self.agent: Agent = agent_name(
            self.simulator.action_space,
            self.simulator.observation_space
        )
        self.num_episodes = num_episodes
        self.batch_episode = batch_episode
        self.summary = summary
        self.render = render

        self.mean_reward = 10
        self.diagnostic = {
            'current_episode': 0,
            'current_timestep': 0
        }

    def run_episode(self):
        obs = self.simulator.reset()

        observations = [obs]
        rewards = []

        while True:
            if self.render:
                self.simulator.render()

            action = self.agent.select_action(obs)
            obs, reward, done, _ = self.simulator.step(action)

            self.diagnostic['current_timestep'] += 1
            rewards.append(reward)
            
            if done:
                break

            observations.append(obs)

        self.diagnostic['current_episode'] += 1

        data = {
            'observations': np.stack(observations),
            'rewards': np.array(rewards)
        }

        return data

    def run(self):
        data = []
        for episode in range(self.num_episodes):
            data.append(self.run_episode())

            if self.diagnostic['current_episode'] % self.batch_episode == 0:
                self.agent.improve(data)
                data = []

        self.simulator.close()

    def evaluate(self):
        pass
