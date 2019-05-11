from itertools import zip_longest

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional
from torch.distributions import Categorical, MultivariateNormal
from torch.optim import Adam


class Actor(nn.Module):
    def __init__(self, obs_space, act_space):
        super().__init__()

        self.epsilon = 0.2
        self.discrete = type(act_space) is gym.spaces.Discrete

        self.fc1 = nn.Linear(obs_space.shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        if self.discrete:
            self.logits = nn.Linear(64, act_space.n)
        else:
            self.means = nn.Linear(64, act_space.shape[0])
            self.eye = torch.eye(act_space.shape[0])

    def forward(self, x, a):
        x = torch.tensor(x, dtype=torch.float)

        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))

        if self.discrete:
            logits = self.logits(x)
            dist = Categorical(logits=logits)
        else:
            means = self.means(x)
            dist = MultivariateNormal(means, self.eye)

        if a is None:
            a = dist.sample().detach().numpy()

        log_prob = dist.log_prob(torch.tensor(a, dtype=torch.float))
        return log_prob, a

    def loss(self, pi_old, obs, act, adv):
        pi_old = torch.tensor(pi_old, dtype=torch.float)
        adv = torch.tensor(adv, dtype=torch.float)

        log_prob, _ = self(obs, act)
        ratio = log_prob.exp() / pi_old

        pi_loss = torch.min(
            ratio * adv,
            torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
        ).mean()

        return -pi_loss


class Critic(nn.Module):
    def __init__(self, obs_space):
        super().__init__()

        self.fc1 = nn.Linear(obs_space.shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)

        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def loss(self, obs, returns):
        returns = torch.tensor(returns, dtype=torch.float)
        values = self(obs).view(-1)
        return functional.smooth_l1_loss(values, returns)


class PPOAgent(nn.Module):
    def __init__(self, obs_space, act_space, logger, num_epochs=10, minibatch_size=64, epsilon=0.2):
        nn.Module.__init__(self)

        self.obs_space = obs_space
        self.act_space = act_space
        self.logger = logger

        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.epsilon = epsilon
        self.discrete = type(act_space) is gym.spaces.Discrete

        self.actor = Actor(obs_space, act_space)
        self.critic = Critic(obs_space)
        self.actor_optimizer = Adam(self.actor.parameters())
        self.critic_optimizer = Adam(self.critic.parameters())

    def forward(self, obs, act):
        log_prob, act = self.actor(obs, act)
        value = self.critic(obs)
        return log_prob, act, value.detach().numpy().reshape(-1)

    def save(self):
        torch.save(self.state_dict(), "model.pt")

    def load(self):
        self.load_state_dict(torch.load("model.pt"))

    def update(self, buffer):
        obs, act, ret, adv = buffer.get()

        # Flattens the input data
        obs = obs.reshape(obs.shape[0] * obs.shape[1], -1)
        act = act.reshape(-1) if self.discrete else act.reshape(act.shape[0] * act.shape[1], -1)
        ret = ret.reshape(-1)
        adv = adv.reshape(-1)

        # Used for visualizations
        actor_loss_buf = []
        critic_loss_buf = []

        # Computes pi_theta_old
        log_prob, _, _ = self(obs, act)
        pi_old = log_prob.exp().detach().numpy()

        # Iterates over the whole dataset num_epochs times
        for _ in range(self.num_epochs):
            # Generates a random indexing of the data
            idx = np.random.choice(np.arange(len(obs)), len(obs), replace=False)

            # Iterates over the data by sequence of size minibatch_size
            for batch_idx in zip_longest(*[iter(idx)] * self.minibatch_size):
                batch_idx = list(batch_idx)  # Converts batch_idx from a tuple to a list

                critic_loss = self.critic.loss(obs[batch_idx], ret[batch_idx])
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                critic_loss_buf.append(critic_loss.detach().numpy().item())

                actor_loss = self.actor.loss(pi_old[batch_idx], obs[batch_idx], act[batch_idx], adv[batch_idx])
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                actor_loss_buf.append(actor_loss.detach().numpy().item())

        self.logger.update(actor_loss=np.mean(actor_loss_buf),
                           critic_loss=np.mean(critic_loss_buf))
