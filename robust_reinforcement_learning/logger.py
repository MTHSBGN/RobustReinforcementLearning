import datetime
import json
import logging
from pathlib import Path
import webbrowser

import numpy as np
import visdom

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


class Logger:
    def __init__(self, logdir: Path):
        self.metrics = {
            "tot_episodes": [],
            "tot_timesteps": []
        }

        self.logdir = logdir
        self.windows = {}

        self.env_name = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        try:
            self.vis = visdom.Visdom(env=self.env_name, raise_exceptions=True)
        except ConnectionError:
            raise Exception("No visdom server detected.")

        webbrowser.open("http://localhost:8097/env/" + self.env_name)

    def update(self, metrics):
        self.log_progress(metrics)

        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []

            self.metrics[key].append(value)

        for k1, k2 in zip(["episodes", "timesteps"], ["tot_episodes", "tot_timesteps"]):
            if k1 in metrics:
                value = metrics[k1] if not len(self.metrics[k2]) else self.metrics[k2][-1] + metrics[k1]
                self.metrics[k2].append(value)

    def plot(self):
        self._plot_reward()
        self._plot_losses()

    def _plot_losses(self):
        for key, value in self.metrics.items():
            if "loss" not in key:
                continue

            if key not in self.windows:
                self.windows[key] = None

            self.windows[key] = self.vis.line(
                value,
                self.metrics["tot_timesteps"],
                win=self.windows[key],
                opts={"title": key.title()}
            )

    def _plot_reward(self):
        mean = np.array(self.metrics["mean_reward"])
        std = np.array(self.metrics["std_reward"])

        x = self.metrics["tot_timesteps"]
        y = self.metrics["mean_reward"]
        y_high = mean + std
        y_low = mean - std

        high = dict(x=x, y=y_high.tolist(), mode="lines", fill=None, type='custom',
                    line=dict(width=0.5, color='rgb(184, 247, 212)'), hoverinfo='none')
        low = dict(x=x, y=y_low.tolist(), mode="lines", fill="tonexty", type='custom',
                   line=dict(width=0.5, color='rgb(184, 247, 212)'), hoverinfo='none')
        mean = dict(x=x, y=y, mode="lines", fill=None, type='custom', name="Mean")
        layout = dict(title="Mean reward",
                      xaxis={"title": "Timsteps"},
                      yaxis={"title": "Reward", "range": [-150, 305]},
                      showlegend=False)
        self.vis._send({'data': [high, low, mean], "layout": layout, 'win': 'mywin'})

    def log_progress(self, metrics):
        logging.info(metrics)
        with Path(self.logdir, "progress.json").open("a+") as f:
            json.dump(metrics, f, cls=NumpyEncoder)
            f.write("\n")
