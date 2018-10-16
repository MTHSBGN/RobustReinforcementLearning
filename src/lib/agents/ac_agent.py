import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from lib.agents.agent import Agent


class ActorCriticAgent(Agent, nn.Module):
    def __init__(self, action_space, observation_space):
        Agent.__init__(self, action_space, observation_space)
        nn.Module.__init__(self)

        self.fc1 = nn.Linear(self.observation_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        self.logits = nn.Linear(64, len(self.actions))
        self.value = nn.Linear(64, 1)

        self.optimizer = optim.RMSprop(self.parameters(), 0.0005)

        self.log_probs = []
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

    def improve(self, rewards):
        cum_reward = 0
        for i in reversed(range(len(rewards))):
            cum_reward += rewards[i]
            rewards[i] = cum_reward

        self.values = torch.cat(self.values)
        rewards = torch.Tensor(rewards)
        advantage = self.values - rewards

        loss = torch.sum(torch.stack(self.log_probs) * advantage)
        loss += nn.functional.smooth_l1_loss(self.values, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.values = []
