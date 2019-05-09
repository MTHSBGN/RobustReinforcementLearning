from itertools import zip_longest

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal
from torch.optim import Adam

import gc

from robust_reinforcement_learning.agents.agent import Agent


class PPOAgent(Agent, nn.Module):
    def __init__(self, obs_space, act_space, num_epochs=10, minibatch_size=32, epsilon=0.2):
        Agent.__init__(self, obs_space, act_space)
        nn.Module.__init__(self)

        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.epsilon = epsilon
        self.discrete = type(act_space) is gym.spaces.Discrete

        self.value = nn.Sequential(
            nn.Linear(obs_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        if self.discrete:
            self.logits = nn.Sequential(
                nn.Linear(obs_space.shape[0], 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            )
        else:
            self.means = nn.Sequential(
                nn.Linear(obs_space.shape[0], 256),
                nn.ReLU(),
                nn.Linear(256, self.act_space.shape[0])
            )

        self.optimizer = Adam(self.parameters(), lr=0.0003)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)

        if self.discrete:
            logits = self.logits(x)
            dist = Categorical(logits=logits)
        else:
            means = self.means(x)
            dist = MultivariateNormal(means, torch.eye(self.act_space.shape[0]))

        value = self.value(x)
        return dist, value

    def get_actions(self, observations, extra_returns=None):
        dist, value = self(observations)
        value = value.detach().numpy().reshape(-1)
        action = dist.sample().numpy()
        if extra_returns and "value" in extra_returns:
            return action, value

        return action

    def save(self):
        torch.save(self.state_dict(), "model.pt")

    def load(self):
        self.load_state_dict(torch.load("model.pt"))

    def update(self, buffer):
        obs, act, ret, adv = buffer.get()

        # Flattens the input data
        obs = obs.reshape(obs.shape[0] * obs.shape[1], -1)
        if self.discrete:
            act = act.reshape(-1)
        else:
            act = act.reshape(act.shape[0] * act.shape[1], -1)
        ret = ret.reshape(-1)
        adv = adv.reshape(-1)

        # Computes pi_theta_old
        dist, values = self(obs)
        pi_old = dist.log_prob(torch.tensor(act, dtype=torch.float)).exp().detach().numpy()

        # Iterates over the whole dataset num_epochs times
        for _ in range(self.num_epochs):
            # Generates a random indexing of the data
            idx = np.random.choice(np.arange(len(obs)), len(obs), replace=False)

            # Iterates over the data by sequence of size minibatch_size
            for batch_idx in zip_longest(*[iter(idx)] * self.minibatch_size):
                batch_idx = list(batch_idx)  # Converts batch_idx from a tuple to a list

                # Computes r_t
                dist, values = self(obs[batch_idx])
                pi_new = dist.log_prob(torch.tensor(act[batch_idx], dtype=torch.float)).exp()
                ratio = pi_new / torch.tensor(pi_old[batch_idx])

                # Computes L_CLIP
                pi_loss = torch.min(
                    ratio * torch.tensor(adv[batch_idx], dtype=torch.float),
                    torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * torch.tensor(adv[batch_idx], dtype=torch.float)
                ).mean()

                # Computes L_VF
                value_loss = nn.functional.smooth_l1_loss(values.view(-1), torch.tensor(ret[batch_idx], dtype=torch.float))

                # Negative pi_loss to perform gradient ascent
                loss = -pi_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
