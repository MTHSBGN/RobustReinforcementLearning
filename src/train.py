import argparse
import datetime
import json
import os
import shutil

import numpy as np

from vec_env import SubprocVecEnv

parser = argparse.ArgumentParser(description='Train a RL agent.')
parser.add_argument('config', type=str, help="Path to a configuration file")


def train(config, directory):
    for num_training in range(config["training"]["num_training"]):
        print("Model {}/{}".format(num_training + 1, config["training"]["num_training"]))
        env_name = config["environment"]["name"]
        EnvClass = getattr(__import__('environments', fromlist=[env_name]), env_name)
        envs = SubprocVecEnv([lambda: EnvClass(config["environment"], True)
                              for _ in range(config["environment"]["num_envs"])])

        # Creates the agent
        agent_name = config["agent"]["name"]
        AgentClass = getattr(__import__('agents', fromlist=[agent_name]), agent_name)
        agent = AgentClass(envs.observation_space.shape[0], envs.action_space.n, 256, config)

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

        # Saves model's weight
        os.makedirs(directory + "/model_" + str(num_training + 1).zfill(2))
        agent.save(directory + "/model_" + str(num_training + 1).zfill(2) + "/model")
        envs.close()


if __name__ == "__main__":
    args = parser.parse_args()

    # Parses the config file
    with open(args.config) as f:
        config_file = json.load(f)

    # Creates the result directory
    path = "experiments/" + datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + "/"
    os.makedirs(path)

    # Moves the config file in the result directory
    shutil.copyfile(args.config, path + "config.json")

    train(config_file, path)
