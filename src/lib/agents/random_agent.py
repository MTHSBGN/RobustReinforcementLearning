import random

from lib.agents.agent import Agent


class RandomAgent(Agent):
    def select_action(self, observation):
        return random.randint(0, 1)

    def improve(self, rewards):
        pass
