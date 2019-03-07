import json

import numpy as np

from utils import rollouts


def evaluate(config, directory):
    env_name = config["environment"]["name"]
    EnvClass = getattr(__import__('environments', fromlist=[env_name]), env_name)
    env_true = EnvClass(config["environment"], True)
    env_false = EnvClass(config["environment"], False)

    # Creates the agent
    agent_name = config["agent"]["name"]
    AgentClass = getattr(__import__('agents', fromlist=[agent_name]), agent_name)
    agent = AgentClass(env_true.observation_space.shape[0], env_true.action_space.n, 256, config)

    data = {}

    for num_training in range(config["training"]["num_training"]):
        print("Model {}/{}".format(num_training + 1, config["training"]["num_training"]))
        agent.load(directory + "/model_" + str(num_training + 1))
        rewards_true = []
        rewards_false = []
        bounds = tuple(config["environment"]["nuisance"]["bounds"])
        for nuisance in np.linspace(bounds[0], bounds[1], 10 * (bounds[1] - bounds[0]) + 1):
            nuisance = np.round(nuisance, 1)
            print("{}/{}".format(nuisance, bounds[1]))
            rewards_true.append(rollouts(agent, env_true, nuisance, 100))
            rewards_false.append(rollouts(agent, env_false, nuisance, 100))

        data["model_{}_true".format(num_training + 1)] = np.array(rewards_true)
        data["model_{}_false".format(num_training + 1)] = np.array(rewards_false)

    np.savez(directory + "/data", **data)
