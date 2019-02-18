import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agents import Agent


class AdversaryActorCriticAgent(Agent):
    def __init__(self, num_inputs, num_outputs, hidden_size, config=None):
        super().__init__(config)

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )

        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value

    def select_actions(self, x):
        x = torch.FloatTensor(x)

        probs = self.actor(x)
        dist = Categorical(probs)

        return dist.sample().detach().numpy()

    def step(self, observations, actions, rewards, masks, last_observation):
        num_trajectories = self.config["environment"]["num_envs"]
        trajectories = [observations[i::num_trajectories] for i in range(num_trajectories)]

        observations = torch.FloatTensor(observations)
        actions = torch.FloatTensor(actions)
        returns = torch.FloatTensor(self._compute_returns(last_observation, rewards, masks)).squeeze(1)

        dist, values = self(observations)
        values = torch.squeeze(values)
        log_probs = dist.log_prob(actions)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _compute_returns(self, last_observations, rewards, masks, gamma=0.99):
        _, values = self(torch.FloatTensor(last_observations))
        R = values.detach().numpy()
        returns = []
        for step in reversed(range(len(rewards))):
            R = np.expand_dims(rewards[step], 1) + gamma * R * np.expand_dims(masks[step], 1)
            returns.insert(0, R)
        return np.concatenate(returns)
