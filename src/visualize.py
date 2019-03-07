import matplotlib
import matplotlib.pyplot as plt

import numpy as np

matplotlib.rcParams['figure.dpi'] = 200


def visualize(config, directory):
    data = np.load(directory + "/data.npz")
    for key in data.files:
        if "true" in key:
            plt.figure(1)
        else:
            plt.figure(2)

        rewards = data[key].transpose()

        # for i in range(rewards.shape[0]):
        #     y = rewards[i, :]
        #     plt.plot(y, color="b", alpha=0.02)

        bounds = tuple(config["environment"]["nuisance"]["bounds"])
        x = list(range(len(np.linspace(bounds[0], bounds[1], 10 * (bounds[1] - bounds[0]) + 1))))
        plt.errorbar(x, np.mean(rewards, axis=0), np.std(rewards, axis=0), fmt='-o', label=str.title(key))

        plt.xticks(list(range(rewards.shape[1])),
                   [str(np.round(x, 1)) for x in np.linspace(bounds[0], bounds[1], 10 * (bounds[1] - bounds[0]) + 1)])
        plt.title("Nuisance: " + config["environment"]["nuisance"]["name"])
        plt.xlabel("Value of the parameters")
        plt.ylabel("Mean reward (n = 1000)")
        plt.legend(loc=4)

    for fig_num in [1, 2]:
        plt.figure(fig_num)
        path = directory + "/plot_" + "true.png" if fig_num == 1 else directory + "/plot_" + "false.png"
        plt.savefig(path)
