from collections import namedtuple
from typing import List

import gym
import numpy as np

from lib.agents import Agent

Trajectory = namedtuple('Trajectory', ['observations', 'actions', 'rewards'])


class Worker:
    def __init__(self, env_name, agent: Agent, num_episodes):
        self.simulator = gym.make(env_name)

        # Might need to change this to parameters of the model / class of the model
        self.agent = agent
        self.num_episodes = num_episodes

    def run(self) -> List[Trajectory]:
        trajectories = []
        for _ in range(self.num_episodes):
            timesteps = []
            obs = self.simulator.reset()

            while True:
                action = self.agent.select_action(obs)
                next_obs, reward, done, _ = self.simulator.step(action)

                timesteps.append((obs, action, reward))
                obs = next_obs

                if done:
                    break

            observations, actions, rewards = zip(*timesteps)
            trajectories.append(Trajectory(
                np.vstack(observations),
                np.array(actions),
                np.array(rewards)
            ))

        return trajectories
