import datetime

import numpy as np
from PIL import Image

from agents import Agent


def evaluate(agent, env):
    """
    Evaluates the performance of an RL agent in the given environment over 100 rollout

    Parameters
    ----------
    agent: Agent
        An implemantation of the Agent's interface
    env: gym.Env
        An instance of a gym's environment

    Returns
    -------
    double
        The mean cumulative reward
    double
        The standard deviation of the cumulative rewards
    """

    rewards = []

    for _ in range(100):
        observation = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_actions([observation])[0]
            observation, reward, done, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)


def save(agent, envs, n=5):
    """
    Saves the model and a rendering of the agent

    Parameters
    ----------
    agent: Agent
        An implemantation of the Agent's interface
    envs: List(gym.Env)
        An instance of a gym's environment
    n: int
        The number of rollout to perform

    Returns
    -------
    None

    """

    filename = "models/" + agent.__class__.__name__ + datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
    names = ["01", "02"]

    for name, env in zip(names, envs):
        images = _rollout(agent, env, n)
        images[0].save(
            filename + "_" + name + ".gif",
            format="GIF",
            append_images=images[1:],
            save_all=True,
            duration=10,
            loop=0
        )

    agent.save(filename)


def _rollout(agent, env, n):
    frames = []

    for _ in range(n):
        observation = env.reset()
        done = False

        while not done:
            frames.append(env.render(mode="rgb_array"))
            action = agent.select_actions([observation])[0]
            observation, _, done, _ = env.step(action)

    return [Image.fromarray(frame) for frame in frames]
