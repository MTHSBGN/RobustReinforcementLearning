import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical


class PGModel(nn.Module):
    def __init__(self, observation_space, action_space, config):
        super().__init__()
        num_inputs = observation_space.shape[0]
        num_outputs = action_space.n

        self.fc1 = nn.Linear(num_inputs, 256)
        self.fc2 = nn.Linear(256, 256)

        self.logits = nn.Linear(256, num_outputs)
        self.value = nn.Linear(256, 1)

    def forward(self, obs):
        encoding = self.fc1(obs)
        encoding = self.fc2(encoding)

        logits = self.logits(encoding)
        value = self.value(encoding)

        return [logits, value]


class PGLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args):
        obs, actions, advs = args

        logits, values = self.model(obs)
        action_dist = Categorical(logits=logits)

        actions = torch.tensor(actions, dtype=torch.float32)
        log_probs = action_dist.log_prob(actions)

        policy_loss = torch.mean(-log_probs * advs)
        value_loss = F.smooth_l1_loss(torch.squeeze(values), advs)

        return policy_loss + value_loss
