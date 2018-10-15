import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class Agent(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)

        self.logits = nn.Linear(64, 2)
        self.value = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

        self.log_probs = []
        self.rewards = []
        self.values = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        log_probs = self.logits(x)
        value = nn.functional.relu(self.value(x))

        return [log_probs, value]

    def select_action(self, observation):
        log_probs, value = self.forward(torch.Tensor(observation))

        m = Categorical(logits=log_probs)
        action = m.sample()

        self.log_probs.append(m.log_prob(action))
        self.values.append(value)

        return action.numpy()

    def observe(self, reward):
        self.rewards.append(reward)

    def improve(self):
        cum_reward = 0
        for i in reversed(range(len(self.rewards))):
            cum_reward += self.rewards[i]
            self.rewards[i] = cum_reward

        self.values = torch.cat(self.values)
        self.rewards = torch.Tensor(self.rewards)
        advantage = self.values - self.rewards

        loss = torch.sum(torch.stack(self.log_probs) * advantage)
        loss += nn.functional.smooth_l1_loss(self.values, self.rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []
        self.values = []
