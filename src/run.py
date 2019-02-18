import argparse
import datetime
import json
import os
from shutil import copyfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from environments import make_training_environment, make_evaluation_environments
from utils import evaluate

matplotlib.rcParams['figure.dpi'] = 200

parser = argparse.ArgumentParser(description='Train a RL agent.')
parser.add_argument('config', type=str, help="Path to a configuration file")

if __name__ == "__main__":
    args = parser.parse_args()

    # Parses the config file
    with open(args.config) as f:
        config = json.load(f)

    # Displays the configuration used
    print("Using the following configuration:")
    print(json.dumps(config, indent=4))

    # Creates the training environements
    envs = make_training_environment(config["environment"])

    # Creates the agent
    agent_name = config["agent"]["name"]
    AgentClass = getattr(__import__('agents', fromlist=[agent_name]), agent_name)
    agent = AgentClass(envs.observation_space.shape[0], envs.action_space.n, 256, config)  # TODO Add model structure to config

    # Trains the agent for N timesteps
    num_timesteps = config["training"]["num_timesteps"]
    state = envs.reset()
    timestep = 0
    while timestep < num_timesteps:
        observations = []
        actions = []
        rewards = []
        masks = []

        # Collect observations for T timesteps
        for _ in range(config["training"]["step"]):
            observations.append(state)
            action = agent.select_actions(state)
            actions.append(action)
            next_state, reward, done, _ = envs.step(action)
            rewards.append(reward)
            masks.append(1 - done)

            state = next_state
            timestep += 1

        last_observation = state
        agent.step(np.vstack(observations), np.concatenate(actions), rewards, masks, last_observation)

        if timestep % 10000 == 0:
            print("Timestep {}/{}".format(timestep, num_timesteps))

    # Creates the result directory
    path = "experiments/" + datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + "/"
    os.makedirs(path)

    # Saves model's weight
    agent.save(path + "model")

    # Moves the config file in the result directory
    copyfile(args.config, path + "config.json")

    # Starts the evaluation
    print("Evaluating:")
    env_true_val, env_false_val = make_evaluation_environments(config["environment"])
    parameters = config["environment"]["parameters"]

    # Iterates over each nuisance parameter
    for key in parameters.keys():
        low, high = tuple(parameters[key])

        p_values = []
        reward_true_values = []
        reward_false_values = []

        # Iterates over the values of the parameters by step of 0.1
        for val in np.linspace(low, high, (high - low) * 10 + 1):
            p = np.round(val, 1)
            print("{}/{}".format(p, high))
            p_values.append(p)
            reward_true_values.append(evaluate(agent, env_true_val, **{key: p}))
            reward_false_values.append(evaluate(agent, env_false_val, **{key: p}))

        plt.plot(reward_true_values, label="True value")
        plt.plot(reward_false_values, label="False value")
        plt.xticks(list(range(len(p_values))), [str(x) for x in p_values])
        plt.xlabel("Value of the parameters")
        plt.ylabel("Mean reward (n = 1000)")
        plt.legend()
        plt.savefig(path + "results.png")
