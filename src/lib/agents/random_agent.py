import random

from lib.agents.agent import Agent


class RandomAgent(Agent):
    def __init__(self, action_space, observation_space):
        super().__init__(action_space, observation_space)

    def select_action(self, observation):
        index = random.randint(0, len(self.actions) - 1)
        return self.actions[index]

    def improve(self, rewards):
        pass
