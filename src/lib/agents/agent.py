from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def select_action(self, observation):
        pass

    @abstractmethod
    def improve(self, rewards):
        pass
