from itertools import zip_longest

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam, RMSprop

from agents.agent import Agent


class PPOAgent(Agent, nn.Module):
    def __init__(self, obs_space, act_space, num_epochs=10, minibatch_size=32, epsilon=0.2):
        Agent.__init__(self, obs_space, act_space)
        nn.Module.__init__(self)

        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.epsilon = epsilon

        in_dim = obs_space.shape[0]

        self.value = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.logits = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        self.optimizer = Adam(self.parameters(), lr=0.0003)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        logits = self.logits(x)
        value = self.value(x)
        dist = Categorical(logits=logits)
        return dist.sample().numpy(), value.detach().numpy().reshape(1, -1)

    def get_actions(self, observations, extra_returns=None):
        action, value = self(observations)
        if extra_returns and "value" in extra_returns:
            return action, value

        return action

    def update(self, buffer):
        obs, act, ret, adv = buffer.get()

        # Flattens the input data
        obs = torch.tensor(obs.reshape(obs.shape[0] * obs.shape[1], -1), dtype=torch.float)
        act = torch.tensor(act.reshape(-1))
        ret = torch.tensor(ret.reshape(-1), dtype=torch.float)
        adv = torch.tensor(adv.reshape(-1), dtype=torch.float)

        # Computes pi_theta_old
        logits = self.logits(obs)
        pi_old = Categorical(logits=logits).log_prob(act).exp().detach()

        # Iterates over the whole dataset num_epochs times
        for _ in range(self.num_epochs):
            # Generates a random indexing of the data
            idx = np.random.choice(np.arange(len(obs)), len(obs), replace=False)

            # Iterates over the data by sequence of size minibatch_size
            for batch_idx in zip_longest(*[iter(idx)] * self.minibatch_size):
                batch_idx = list(batch_idx)  # Converts batch_idx from a tuple to a list

                # Computes r_t
                logits = self.logits(obs[batch_idx])
                new_pi = Categorical(logits=logits).log_prob(act[batch_idx]).exp()
                ratio = new_pi / pi_old[batch_idx]

                # Computes L_CLIP
                pi_loss = torch.min(
                    ratio * adv[batch_idx],
                    torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv[batch_idx]
                ).mean()

                # Computes L_VF
                values = self.value(obs[batch_idx]).reshape(-1)
                value_loss = nn.functional.smooth_l1_loss(values, ret[batch_idx])

                # Negative pi_loss to perform gradient ascent
                loss = -pi_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
