import numpy as np
import scipy.signal

from robust_reinforcement_learning.vec_env import SubprocVecEnv
from robust_reinforcement_learning.environments import env_creator


class Buffer:
    def __init__(self, agent, config):
        self.agent = agent
        self.envs = SubprocVecEnv([lambda: env_creator(config["env_config"]) for _ in range(config["num_envs"])])
        self.num_step = config["num_steps"]
        self.gamma = config["gamma"]
        self.lam = config["lambda"]

        size = (self.envs.nenvs, self.num_step)
        self.obs_buf = np.zeros(size + self.envs.observation_space.shape)
        self.act_buf = np.zeros(size + self.envs.action_space.shape)
        self.rew_buf = np.zeros(size)
        self.val_buf = np.zeros(size)
        self.ret_buf = np.zeros(size)
        self.adv_buf = np.zeros(size)

        self.cum_rew = np.zeros(self.envs.nenvs)
        self.ep_start = [[0] for _ in range(self.envs.nenvs)]

        self.last_obs = self.envs.reset()

    def collect(self):
        tot_rewards = []
        start_idx = np.zeros(self.envs.nenvs, dtype=np.int)
        obs = self.last_obs
        self.ep_start = [[0] for _ in range(self.envs.nenvs)]

        for idx in range(self.num_step):
            actions, values = self.agent.select_action(obs)
            next_obs, rewards, dones, _ = self.envs.step(actions)

            self.obs_buf[:, idx] = obs
            self.act_buf[:, idx] = actions
            self.rew_buf[:, idx] = rewards
            self.val_buf[:, idx] = values
            self.cum_rew[:] += rewards

            obs = next_obs

            for i in np.argwhere(dones).reshape(-1):
                tot_rewards.append(self.cum_rew[i])
                self.cum_rew[i] = 0
                self.finish_path(i, start_idx[i], idx + 1)
                start_idx[i] = idx + 1
                self.ep_start[i].append(idx + 1)

        self.last_obs = obs

        for i in range(self.envs.nenvs):
            self.finish_path(i, start_idx[i], self.num_step, self.val_buf[i, -1])

        return {
            "episodes": len(tot_rewards),
            "timesteps": self.envs.nenvs * self.num_step,
            "min_reward": np.min(tot_rewards),
            "mean_reward": np.mean(tot_rewards),
            "max_reward": np.max(tot_rewards),
            "std_reward": np.std(tot_rewards)
        }

    def finish_path(self, idx, start, end, last_val=0):
        s = slice(start, end)
        rews = np.append(self.rew_buf[idx, s], last_val)
        vals = np.append(self.val_buf[idx, s], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

        self.ret_buf[idx, s] = discount_cumsum(rews, self.gamma)[:-1]
        self.adv_buf[idx, s] = discount_cumsum(deltas, self.gamma * self.lam)

    def get(self):
        return self.obs_buf, self.act_buf, self.ret_buf, self.adv_buf

    def get_trajectories(self):
        episodes = []
        for i in range(self.envs.nenvs):
            data = {
                "observations": self.obs_buf[i, :],
                "actions": self.act_buf[i, :],
                "length": len(self.obs_buf[i, :])
            }

            if len(self.ep_start[i]) == 1:
                episodes.append(data)
            else:
                for start, end in zip(self.ep_start[i], self.ep_start[i][1:] + [self.num_step]):
                    episodes.append({
                        "observations": data["observations"][start:end],
                        "actions": data["actions"][start:end],
                        "length": len(data["observations"][start:end])
                    })

        return episodes


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
