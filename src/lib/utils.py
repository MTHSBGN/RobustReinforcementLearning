import numpy as np


def discounted_returns(rewards, gamma):
    for reward in rewards:
        R = 0
        index = len(reward) - 1
        for r in reversed(reward):
            reward[index] = r + gamma * R
            R = reward[index]
            index -= 1

    return np.concatenate(rewards)
