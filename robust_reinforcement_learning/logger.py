import datetime
import numpy as np
import time
import visdom
import webbrowser


class Logger:
    def __init__(self):
        self.data = {}
        self.opts = {
            "mean_reward": {
                "ytickmin": -125,
                "ytickmax": 350,
            }
        }
        self.env_name = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

        try:
            self.vis = visdom.Visdom(env=self.env_name, raise_exceptions=True)
        except ConnectionError:
            raise Exception("No visdom server detected.")

        self.windows = {}
        webbrowser.open("http://localhost:8097/env/" + self.env_name)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []
                self.windows[key] = None
                if key not in self.opts:
                    self.opts[key] = {}

            self.data[key].append(value)

    def plot(self, timestep):
        for key, value in self.data.items():
            if len(value) == 0:
                continue

            time.sleep(0.01)
            self.windows[key] = self.vis.line(
                [[value[-1], np.mean(value)]],
                [[timestep, timestep]],
                win=self.windows[key],
                update="append" if self.windows[key] else None,
                opts=dict(
                    width=500,
                    xlabel="Timestep",
                    title=key.replace("_", " ").title(),
                    **self.opts[key]
                )
            )

            if key == "reward":
                self.data[key] = []

            if len(value) >= 10:
                del value[0]
