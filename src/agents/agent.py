import torch.nn as nn


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def select_actions(self, x):
        raise NotImplementedError

    def step(self, observations, actions, rewards, masks, last_observation):
        raise NotImplementedError
