import matplotlib
import matplotlib.pyplot as plt

import numpy as np

plt.style.use('ggplot')

matplotlib.rcParams['figure.dpi'] = 200


def rollouts(agent, env, nuisance, n=1000):
    rewards = []
    actions = []

    for i in range(n):
        state = env.reset(nuisance)
        done = False
        cum_reward = 0
        while not done:
            action = agent.select_actions([state])[0]
            actions.append(action)
            state, reward, done, _ = env.step(action)
            cum_reward += reward

        rewards.append(cum_reward)

    rewards = np.array(rewards)
    actions = np.array(actions)

    _, num_actions = np.unique(actions, return_counts=True)

    return [np.mean(rewards), np.std(rewards)] + num_actions.tolist()


def plot_rewards(rewards, var, out_dir, labels, title):
    plt.figure()

    for mean, var, s in zip(rewards, var, ["True", "False"]):
        plt.errorbar(np.arange(10), mean, var, fmt='-o', label=s + " value")

    plt.ylim(0, 220)
    plt.xticks(np.arange(len(labels)), labels)

    plt.title(title)
    plt.xlabel("Value of the parameters")
    plt.ylabel("Mean reward (n = 1000)")
    plt.legend(loc=4)

    plt.savefig(out_dir)
    plt.close()


def plot_ratio_actions(data, out_dir, labels, title):
    plt.figure()

    for d, s in zip(data, ["True", "False"]):
        ratio = (d[:, 0] - d[:, 1]) / (d[:, 0] + d[:, 1])
        plt.plot(np.arange(10), ratio, marker="o", label=s + " value")

    plt.ylim(-0.2, 0.2)
    plt.xticks(np.arange(len(labels)), labels)

    plt.title(title)
    plt.xlabel("Value of the parameters")
    plt.ylabel("Ratio between # of actions")
    plt.legend()

    plt.savefig(out_dir)
    plt.close()
