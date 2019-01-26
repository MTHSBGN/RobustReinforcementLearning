import argparse
import datetime
import json
import os

import numpy as np

from environments import make_environment

parser = argparse.ArgumentParser(description='Train a RL agent.')
parser.add_argument('config', type=str, help="Path to a configuration file")

if __name__ == "__main__":
    args = parser.parse_args()

    # Parses the config file
    with open(args.config) as f:
        config = json.load(f)

    # Creates the training environements
    envs = make_environment(config["environment"])

    # Creates the agent
    agent_name = config["agent"]["name"]
    AgentClass = getattr(__import__('agents', fromlist=[agent_name]), agent_name)
    agent = AgentClass(envs.observation_space.shape[0], envs.action_space.n, 256)  # TODO Add model structure to config

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
    os.rename(args.config, path + "config.json")
