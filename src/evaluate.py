import argparse
import json

import numpy as np

from utils import rollouts

parser = argparse.ArgumentParser(description='Train a RL agent.')
parser.add_argument('directory', type=str, help="Path to a directory containing models to evaluate")


def evaluate(config, directory):
    env_name = config["environment"]["name"]
    env_class = getattr(__import__('environments', fromlist=[env_name]), env_name)
    env_true = env_class(config["environment"], True)
    env_false = env_class(config["environment"], False)

    # Creates the agent
    agent_name = config["agent"]["name"]
    agent_class = getattr(__import__('agents', fromlist=[agent_name]), agent_name)
    agent = agent_class(env_true.observation_space.shape[0], env_true.action_space.n, 256, config)

    for num_training in range(config["training"]["num_training"]):
        print("Model {}/{}".format(num_training + 1, config["training"]["num_training"]))
        agent.load(directory + "/model_" + str(num_training + 1).zfill(2) + "/model")
        bounds = tuple(config["environment"]["nuisance"]["bounds"])

        data = {
            "true": [],
            "false": []
        }

        for nuisance in np.around(np.linspace(bounds[0], bounds[1], num=10), 1):
            data["true"].append(rollouts(agent, env_true, nuisance))
            data["false"].append(rollouts(agent, env_false, nuisance))

        data["true"] = np.array(data["true"])
        data["false"] = np.array(data["false"])
        np.savez(directory + "/model_" + str(num_training + 1).zfill(2) + "/data", **data)


if __name__ == "__main__":
    args = parser.parse_args()

    # Parses the config file
    with open(args.directory + "/config.json") as f:
        config_file = json.load(f)

    evaluate(config_file, args.directory)
