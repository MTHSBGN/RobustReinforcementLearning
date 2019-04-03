import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils import plot_rewards, plot_ratio_actions

plt.style.use('ggplot')

matplotlib.rcParams['figure.dpi'] = 200

parser = argparse.ArgumentParser(description='Train a RL agent.')
parser.add_argument('directory', type=str, help="Path to a directory containing models to visualize")


def visualize(config, directory):
    bounds = tuple(config["environment"]["nuisance"]["bounds"])
    labels = [str(x) for x in np.around(np.linspace(bounds[0], bounds[1], num=10), 1)]
    rewards_true, rewards_false, var_true, var_false, actions_true, actions_false = [], [], [], [], [], []

    for num_training in range(config["training"]["num_training"]):
        data = np.load(directory + "/model_" + str(num_training + 1).zfill(2) + "/data.npz")

        data_true = data["true"]
        data_false = data["false"]

        rewards_true.append(data_true[:, 0])
        rewards_false.append(data_false[:, 0])
        var_true.append(data_true[:, 1])
        var_false.append(data_false[:, 1])
        actions_true.append(data_true[:, 2:4])
        actions_false.append(data_false[:, 2:4])

        plot_rewards(
            [data_true[:, 0], data_false[:, 0]],
            [data_true[:, 1], data_false[:, 1]],
            directory + "/model_" + str(num_training + 1).zfill(2) + "/rewards.png",
            labels,
            "Nuisance: " + config["environment"]["nuisance"]["name"]
        )

        plot_ratio_actions(
            [data_true[:, 2:4], data_false[:, 2:4]],
            directory + "/model_" + str(num_training + 1).zfill(2) + "/actions.png",
            labels,
            "Nuisance: " + config["environment"]["nuisance"]["name"]
        )

    plot_rewards(
        [np.mean(np.vstack(rewards_true), axis=0), np.mean(np.vstack(rewards_false), axis=0)],
        [np.mean(np.vstack(var_true), axis=0), np.mean(np.vstack(var_false), axis=0)],
        directory + "/rewards.png",
        labels,
        "Nuisance: " + config["environment"]["nuisance"]["name"]
    )

    plot_ratio_actions(
        [np.mean(np.stack(actions_true), axis=0), np.mean(np.stack(actions_false), axis=0)],
        directory + "/actions.png",
        labels,
        "Nuisance: " + config["environment"]["nuisance"]["name"]
    )


if __name__ == "__main__":
    args = parser.parse_args()

    # Parses the config file
    with open(args.directory + "/config.json") as f:
        config_file = json.load(f)

    visualize(config_file, args.directory)
