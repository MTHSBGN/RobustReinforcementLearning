from itertools import zip_longest

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional
from torch.distributions import MultivariateNormal
from torch.optim import Adam


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, config):
        super().__init__()

        self.epsilon = config["epsilon"]

        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        self.means = nn.Linear(64, act_dim)
        self.eye = torch.eye(act_dim)

    def forward(self, x, a):
        x = torch.tensor(x, dtype=torch.float)

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

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
    def __init__(self, obs_dim):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, 64)
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


class Adversary(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_lens = 100

        self.lstm1 = nn.LSTM(num_inputs, hidden_dim)
        self.nuisance_out = nn.Linear(hidden_dim, num_outputs)

    def forward(self, inputs):
        out, _ = self.lstm1(inputs)
        nuisance = self.nuisance_out(out[-1])
        return nuisance.view(-1)

    def loss(self, episodes, actor):
        print("Num episodes: ", len(episodes))
        out = [self.split_episode(episode, actor) for episode in episodes]

        sequences = []
        nuisance_target = []
        for seq, nuisance in out:
            if type(seq) is list:
                continue
            sequences.append(seq)
            nuisance_target.append(nuisance)


        sequences = torch.cat(sequences, 1)
        print("Num sequences: ", sequences.shape[1])

        nuisance_target = torch.cat(nuisance_target)
        nuisance_pred = self(sequences)
        return nn.functional.mse_loss(nuisance_pred, nuisance_target)

    def split_episode(self, episode, actor):
        if episode["length"] < self.seq_lens:
            return [], []

        obs = episode["observations"]
        actions = episode["actions"]
        log_prob, _ = actor(obs, actions)

        nuisance_target = obs[:, 0]
        obs = torch.tensor(obs[:, 1:], dtype=torch.float)
        obs = torch.cat([obs, log_prob.view(-1, 1)], 1)

        sequences = torch.split(obs.unsqueeze(1), self.seq_lens)
        if sequences[-1].shape[0] < self.seq_lens:
            sequences = sequences[:-1]

        return torch.cat(sequences, 1), torch.tensor(nuisance_target[:len(sequences)], dtype=torch.float)


class PPO(nn.Module):
    def __init__(self, obs_space, act_space, config):
        nn.Module.__init__(self)
        self.config = config

        self.num_epochs = config["num_epochs"]
        self.minibatch_size = config["minibatch_size"]

        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]

        self.actor = Actor(obs_dim, act_dim, config)
        self.critic = Critic(obs_dim)

        if config["adversary"]:
            self.adversary = Adversary(obs_dim, 1, 256)

        self.policy_optimizer = Adam(self.parameters(), lr=0.001)
        self.adversary_optimizer = Adam(self.adversary.parameters(), lr=0.001)

    def forward(self, obs, act):
        log_prob, act = self.actor(obs, act)
        value = self.critic(obs)
        return log_prob, act, value.detach().numpy().reshape(-1)

    def save(self, path):
        torch.save(self.state_dict(), path + "/model.pt")

    def load(self, checkpoint):
        self.load_state_dict(torch.load(checkpoint))

    def update(self, buffer):
        obs, act, ret, adv = buffer.get()

        # Flattens the input data
        obs = obs.reshape(obs.shape[0] * obs.shape[1], -1)
        act = act.reshape(act.shape[0] * act.shape[1], -1)
        ret = ret.reshape(-1)
        adv = adv.reshape(-1)

        # Computes pi_theta_old
        log_prob, _, _ = self(obs, act)
        pi_old = log_prob.exp().detach().numpy()

        # Used for metrics
        actor_buf = []
        critic_buf = []
        loss_buf = []

        # Iterates over the whole dataset num_epochs times
        for epoch in range(self.num_epochs):
            print("Epoch: ", epoch)
            # Generates a random indexing of the data
            idx = np.random.choice(np.arange(len(obs)), len(obs), replace=False)

            # Iterates over the data by sequence of size minibatch_size
            for batch_idx in zip_longest(*[iter(idx)] * self.minibatch_size):
                batch_idx = list(batch_idx)  # Converts batch_idx from a tuple to a list

                actor_loss = self.actor.loss(pi_old[batch_idx], obs[batch_idx], act[batch_idx], adv[batch_idx])
                critic_loss = self.critic.loss(obs[batch_idx], ret[batch_idx])
                loss = actor_loss + critic_loss

                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()

                actor_buf.append(actor_loss.detach().numpy())
                critic_buf.append(critic_loss.detach().numpy())
                loss_buf.append(loss.detach().numpy())

        if self.config["adversary"]:
            adversary_loss = -self.adversary.loss(buffer.get_trajectories(), self.actor)
            self.adversary_optimizer.zero_grad()
            adversary_loss.backward()
            self.adversary_optimizer.step()

        return {
            "actor_loss": np.mean(actor_buf),
            "critic_loss": np.mean(critic_buf),
            "adversary_loss": -adversary_loss.detach().numpy() if self.config["adversary"] else 0,
            "loss": np.mean(loss_buf)
        }
