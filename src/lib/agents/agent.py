from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def select_action(self, observation):
        pass

    @abstractmethod
    def improve(self, rewards):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
