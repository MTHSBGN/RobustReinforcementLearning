import torch
import torch.nn as nn


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.models_path = "models"

    def forward(self, x):
        raise NotImplementedError

    def select_actions(self, x):
        raise NotImplementedError

    def step(self, observations, actions, rewards, masks, last_observation):
        raise NotImplementedError

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        path = "{}/{}".format(self.models_path, filename)
        self.load_state_dict(torch.load(path))
