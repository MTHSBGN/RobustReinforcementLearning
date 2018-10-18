import gym
import numpy as np

from lib.agents import Agent

from collections import deque


class Experiment:
    def __init__(self, agent_name, env_name, num_episodes=1000, batch_episode=10, summary=False, render=False):
        self.simulator = gym.make(env_name)
        self.agent: Agent = agent_name(
            self.simulator.action_space,
            self.simulator.observation_space
        )
        self.num_episodes = num_episodes
        self.batch_episode = batch_episode
        self.summary = summary
        self.render = render

        self.diagnostic = {
            'episode': 0,
            'timestep': 0,
            'max_reward': 0,
            'min_reward': 9999999,
            'rewards': deque(maxlen=100)
        }

        self.__init_stored_data()

    def run_episode(self):
        obs = self.simulator.reset()

        actions = []
        observations = [obs]
        rewards = []

        while True:
            if self.render:
                self.simulator.render()

            action = self.agent.select_action(obs)
            obs, reward, done, _ = self.simulator.step(action)

            self.diagnostic['timestep'] += 1
            actions.append(action)
            rewards.append(reward)

            if done:
                break

            observations.append(obs)

        self.__update_diagnostic(sum(rewards))

        self.stored_data['actions'].append(np.array(actions))
        self.stored_data['observations'].append(np.stack(observations))
        self.stored_data['rewards'].append(np.array(rewards))

    def run(self):
        for episode in range(self.num_episodes):
            self.run_episode()

            if self.diagnostic['episode'] % self.batch_episode == 0:
                if self.summary:
                    self.__summary()
                self.agent.improve(self.stored_data)
                self.__init_stored_data()

        self.agent.save('test.pt')

        self.simulator.close()

    def evaluate(self, model_path):
        self.agent.load(model_path)

        for _ in range(100):
            obs = self.simulator.reset()

            while True:
                self.simulator.render()

                action = self.agent.select_action(obs)
                obs, _, done, _ = self.simulator.step(action)

                if done:
                    break

        self.simulator.close()

    def __summary(self):
        print("##################")
        print("Episode {} ({})".format(
            self.diagnostic['episode'],
            self.diagnostic['timestep']
        ))
        print('Max: ', self.diagnostic['max_reward'])
        print('Min: ', self.diagnostic['min_reward'])
        print('Mean: ', np.mean(self.diagnostic['rewards']))
        print('Std: ', np.std(self.diagnostic['rewards']))

    def __update_diagnostic(self, reward):
        self.diagnostic['episode'] += 1

        if reward > self.diagnostic['max_reward']:
            self.diagnostic['max_reward'] = reward
        elif reward < self.diagnostic['min_reward']:
            self.diagnostic['min_reward'] = reward

        self.diagnostic['rewards'].append(reward)

    def __init_stored_data(self):
        self.stored_data = {
            'actions': [],
            'observations': [],
            'rewards': []
        }
