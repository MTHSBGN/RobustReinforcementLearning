import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
from copy import deepcopy

from lib.agents.agent import Agent
from lib.utils import discounted_returns

GAMMA = 0.99
EPOCH = 10
BATCH_SIZE = 64
EPSILON = 0.2


class PPOAgent(Agent, nn.Module):
    def __init__(self, action_space, observation_space):
        Agent.__init__(self, action_space, observation_space)
        nn.Module.__init__(self)

        self.policy_fc1 = nn.Linear(self.observation_dim, 64)
        self.policy_fc2 = nn.Linear(64, 64)
        self.policy_out = nn.Linear(64, len(self.actions))

        self.value_fc1 = nn.Linear(self.observation_dim, 64)
        self.value_fc2 = nn.Linear(64, 64)
        self.value_out = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(), 3 * 10**-4)

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

        old_agent = deepcopy(self)

        obs = torch.Tensor(obs)
        actions = torch.Tensor(actions)

        for epoch in range(EPOCH):
            samples_used = 0
            while samples_used < len(actions):
                samples_used += BATCH_SIZE
                index = np.random.randint(
                    0,
                    high=len(actions),
                    size=BATCH_SIZE
                )

                batch_obs = obs[index]
                batch_actions = actions[index]
                batch_advantage = advantage[index]

                l_p, v = self.forward(
                    batch_obs,
                    actions=batch_actions
                )
                l_p_old, v_old = old_agent.forward(
                    batch_obs,
                    actions=batch_actions
                )

                ratios = torch.exp(l_p) / torch.exp(l_p_old)
                clipped = torch.clamp(ratios, 1 - EPSILON, 1 + EPSILON)

                policy_loss = torch.mean(torch.min(
                    ratios * batch_advantage,
                    clipped * batch_advantage)
                )

                value_loss = nn.functional.smooth_l1_loss(
                    values[index],
                    rewards[index]
                )

                loss = policy_loss + value_loss
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
