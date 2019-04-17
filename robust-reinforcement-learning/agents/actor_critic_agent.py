import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import RMSprop

from agents.agent import Agent


class ActorCriticAgent(Agent, nn.Module):
    def __init__(self, obs_space, act_space):
        Agent.__init__(self, obs_space, act_space)
        nn.Module.__init__(self)

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

        self.optimizer = RMSprop(self.parameters(), lr=0.0003)

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

        obs = torch.tensor(obs.reshape(obs.shape[0] * obs.shape[1], -1), dtype=torch.float)
        act = torch.tensor(act.reshape(-1))
        ret = torch.tensor(ret.reshape(-1), dtype=torch.float)
        adv = torch.tensor(adv.reshape(-1), dtype=torch.float)

        logits = self.logits(obs)
        log_p = Categorical(logits=logits).log_prob(act)
        values = self.value(obs).reshape(-1)

        pi_loss = torch.mean(-log_p * adv)
        value_loss = nn.functional.smooth_l1_loss(values, ret)
        loss = pi_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
