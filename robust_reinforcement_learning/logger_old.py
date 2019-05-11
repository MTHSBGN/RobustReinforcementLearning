from collections import deque
from datetime import datetime
import numpy as np
import visdom
import webbrowser
import time


class Logger:
    def __init__(self):
        self.env_name = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

        try:
            self.vis = visdom.Visdom(env=self.env_name, raise_exceptions=True)
        except ConnectionError:
            raise Exception("No visdom server detected. Run the command \"visdom\" in your CLI to start it.")

        webbrowser.open("http://localhost:8097/env/" + self.env_name)
        self.reward_win = None
        self.actor_loss_win = None
        self.critic_loss_win = None
        self.value_win = None

        self.actor_loss_buf = deque(maxlen=10)
        self.critic_loss_buf = deque(maxlen=10)
        self.value_buf = deque(maxlen=10)

    def update_loss_buffers(self, actor_value, critic_value):
        self.actor_loss_buf.append(actor_value)
        self.critic_loss_buf.append(critic_value)

    def update_value_buffer(self, value):
        self.value_buf.append(value)

    def plot_reward(self, reward, step, goal):
        time.sleep(0.01)
        self.reward_win = self.vis.line(
            [[reward, goal]],
            [[step, step]],
            win=self.reward_win,
            update="append" if self.reward_win else None,
            opts=dict(
                xlabel="Timestep",
                ylabel="Reward",
                title="Mean reward",
                width=500
            )
        )

    def plot_losses(self, step):
        time.sleep(0.01)
        self.actor_loss_win = self.vis.line(
            [[self.actor_loss_buf[len(self.actor_loss_buf) - 1], np.mean(self.actor_loss_buf)]],
            [[step, step]],
            win=self.actor_loss_win,
            update="append" if self.actor_loss_win else None,
            opts=dict(
                xlabel="Timestep",
                title="Actor Loss",
                width=500
            )
        )

        time.sleep(0.01)
        self.critic_loss_win = self.vis.line(
            [[self.critic_loss_buf[len(self.critic_loss_buf) - 1], np.mean(self.critic_loss_buf)]],
            [[step, step]],
            win=self.critic_loss_win,
            update="append" if self.critic_loss_win else None,
            opts=dict(
                xlabel="Timestep",
                title="Critic Loss",
                width=500
            )
        )

    def plot_value(self, step):
        time.sleep(0.01)
        self.value_win = self.vis.line(
            [[self.value_buf[len(self.value_buf) - 1], np.mean(self.value_buf)]],
            [[step, step]],
            win=self.value_win,
            update="append" if self.value_win else None,
            opts=dict(
                xlabel="Timestep",
                title="Value estimate",
                width=500
            )
        )
