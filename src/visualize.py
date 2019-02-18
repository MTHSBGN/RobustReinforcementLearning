import argparse
import json

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

matplotlib.rcParams['figure.dpi'] = 200

parser = argparse.ArgumentParser(description='Train a RL agent.')
parser.add_argument('config', type=str, help="Path to a configuration file")

if __name__ == "__main__":
    args = parser.parse_args()

    # Parses the config file
    with open(args.config + "/config.json") as f:
        config = json.load(f)

    data = np.load(args.config + "/data.npz")
    for key in data.files:
        rewards = data[key].transpose()

        for i in range(rewards.shape[0]):
            y = rewards[i, :]
            plt.plot(y, color="b", alpha=0.02)

        plt.plot(np.mean(rewards, axis=0), color="r", label=str.title(key))

        bounds = tuple(config["environment"]["nuisance"]["bounds"])
        plt.xticks(list(range(rewards.shape[1])),
                   [str(np.round(x, 1)) for x in np.linspace(bounds[0], bounds[1], 10 * (bounds[1] - bounds[0]) + 1)])
        plt.title("Nuisance: " + config["environment"]["nuisance"]["name"])
        plt.xlabel("Value of the parameters")
        plt.ylabel("Mean reward (n = 1000)")
        plt.legend(loc=4)

    plt.show()
