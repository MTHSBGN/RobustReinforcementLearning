import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

from lib.agents.agent import Agent
from lib.utils import discounted_returns

GAMMA = 0.99


class ActorCriticAgent(Agent, nn.Module):
    def __init__(self):
        super().__init__()

        self.policy_fc1 = nn.Linear(4, 64)
        self.policy_fc2 = nn.Linear(64, 64)
        self.policy_out = nn.Linear(64, 2)

        self.value_fc1 = nn.Linear(4, 64)
        self.value_fc2 = nn.Linear(64, 64)
        self.value_out = nn.Linear(64, 1)

        self.optimizer = optim.RMSprop(self.parameters(), 0.001)

    def forward(self, x, actions=None):
        log_probs = self.policy_fc1(x)
        log_probs = self.policy_fc2(log_probs)
        log_probs = self.policy_out(log_probs)

        value = self.value_fc1(x)
        value = self.value_fc2(value)
        value = self.value_out(value)

        if actions is not None:
            distributions = Categorical(logits=log_probs)
            log_probs = distributions.log_prob(actions)

        return [log_probs, value]

    def select_action(self, observation):
        log_probs, value = self.forward(torch.Tensor(observation))

        m = Categorical(logits=log_probs)
        action = m.sample()

        return action.numpy()

    def improve(self, data):
        obs = np.vstack(data['observations'])
        actions = np.concatenate(data['actions'])
        log_probs, values = self.forward(
            torch.Tensor(obs),
            actions=torch.Tensor(actions)
        )

        cum_rewards = discounted_returns(data['rewards'], GAMMA)

        values = torch.squeeze(values)
        rewards = torch.Tensor(cum_rewards)
        advantage = rewards - values

        policy_loss = torch.mean(-log_probs * advantage)
        value_loss = nn.functional.smooth_l1_loss(values, rewards)

        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
