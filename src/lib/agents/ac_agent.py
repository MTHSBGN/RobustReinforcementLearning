import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

from lib.agents.agent import Agent


class ActorCriticAgent(Agent, nn.Module):
    def __init__(self, action_space, observation_space):
        Agent.__init__(self, action_space, observation_space)
        nn.Module.__init__(self)

        self.policy_fc1 = nn.Linear(self.observation_dim, 64)
        self.policy_fc2 = nn.Linear(64, 64)
        self.policy_out = nn.Linear(64, len(self.actions))

        self.value_fc1 = nn.Linear(self.observation_dim, 64)
        self.value_fc2 = nn.Linear(64, 64)
        self.value_out = nn.Linear(64, 1)

        self.optimizer = optim.RMSprop(self.parameters(), 0.001)

        self.log_probs = []

    def forward(self, x):
        log_probs = self.policy_fc1(x)
        log_probs = self.policy_fc2(log_probs)
        log_probs = self.policy_out(log_probs)

        value = self.value_fc1(x)
        value = self.value_fc2(value)
        value = self.value_out(value)

        return [log_probs, value]

    def select_action(self, observation):
        log_probs, value = self.forward(torch.Tensor(observation))

        m = Categorical(logits=log_probs)
        action = m.sample()

        self.log_probs.append(m.log_prob(action))

        return action.numpy()

    def improve(self, data):
        obs = np.vstack([d['observations'] for d in data])
        _, values = self.forward(torch.Tensor(obs))

        cum_rewards = self.__compute_cumulative_rewards(
            [d['rewards'] for d in data])

        values = torch.squeeze(values)
        rewards = torch.Tensor(cum_rewards)
        advantage = rewards - values

        policy_loss = torch.mean(-torch.stack(self.log_probs) * advantage)
        value_loss = nn.functional.smooth_l1_loss(values, rewards)

        # print("{}: {}".format(policy_loss, value_loss))

        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []

    def __compute_cumulative_rewards(self, rewards):
        cum_rewards = np.empty(sum(len(x) for x in rewards))
        index = len(cum_rewards) - 1
        for reward in rewards:
            cum_sum = 0
            for i in reversed(range(len(reward))):
                cum_sum += reward[i]
                cum_rewards[index] = cum_sum
                index -= 1

        return cum_rewards

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
