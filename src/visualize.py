import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['figure.dpi'] = 200

parser = argparse.ArgumentParser(description='Train a RL agent.')
parser.add_argument('directory', type=str, help="Path to a directory containing models to visualize")


def visualize(config, directory):
    bounds = tuple(config["environment"]["nuisance"]["bounds"])
    x = list(range(10))

    for num_training in range(config["training"]["num_training"]):
        plt.figure()
        data = np.load(directory + "/model_" + str(num_training + 1).zfill(2) + "/data.npz")

        data_true = data["true"]
        data_false = data["false"]

        plt.errorbar(x, data_true[:, 0], data_true[:, 1], fmt='-o', label="True value")
        plt.errorbar(x, data_false[:, 0], data_false[:, 1], fmt='-o', label="False value")

        labels = [str(x) for x in np.around(np.linspace(bounds[0], bounds[1], num=10), 1)]
        plt.xticks(list(range(len(labels))), labels)

        plt.ylim(0, 220)

        plt.title("Nuisance: " + config["environment"]["nuisance"]["name"])
        plt.xlabel("Value of the parameters")
        plt.ylabel("Mean reward (n = 1000)")
        plt.legend(loc=4)

        plt.savefig(directory + "/model_" + str(num_training + 1).zfill(2) + "/rewards.png")


if __name__ == "__main__":
    args = parser.parse_args()

    # Parses the config file
    with open(args.directory + "/config.json") as f:
        config_file = json.load(f)

    visualize(config_file, args.directory)
