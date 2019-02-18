import torch
import torch.nn as nn


class Agent(nn.Module):
    def __init__(self, config):
        super(Agent, self).__init__()
        self.config = config

    def forward(self, x):
        raise NotImplementedError

    def select_actions(self, x):
        raise NotImplementedError

    def step(self, observations, actions, rewards, masks, last_observation):
        raise NotImplementedError

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
