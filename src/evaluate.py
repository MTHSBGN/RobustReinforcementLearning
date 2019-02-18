import argparse
import json

import numpy as np

parser = argparse.ArgumentParser(description='Train a RL agent.')
parser.add_argument('config', type=str, help="Path to a configuration file")

if __name__ == "__main__":
    args = parser.parse_args()

    # Parses the config file
    with open(args.config + "/config.json") as f:
        config = json.load(f)

    # Displays the configuration used
    print("Using the following configuration:")
    print(json.dumps(config, indent=4))

    env_name = config["environment"]["name"]
    EnvClass = getattr(__import__('environments', fromlist=[env_name]), env_name)
    env = EnvClass(config["environment"], True)

    # Creates the agent
    agent_name = config["agent"]["name"]
    AgentClass = getattr(__import__('agents', fromlist=[agent_name]), agent_name)
    agent = AgentClass(env.observation_space.shape[0], env.action_space.n, 256, config)

    data = {}

    print("Nuisance: " + config["environment"]["nuisance"]["name"])
    for num_training in range(config["training"]["num_training"]):
        print("Model " + str(num_training))
        agent.load(args.config + "/model_" + str(num_training))
        rewards = []
        bounds = tuple(config["environment"]["nuisance"]["bounds"])
        for length in np.linspace(bounds[0], bounds[1], 10 * (bounds[1] - bounds[0]) + 1):
            length = np.round(length, 1)
            print("{}/{}".format(length, bounds[1]))
            current_rewards = []
            for i in range(1000):
                state = env.reset(length)
                done = False
                cum_reward = 0
                while not done:
                    action = agent.select_actions([state])[0]
                    state, reward, done, _ = env.step(action)
                    cum_reward += reward

                current_rewards.append(cum_reward)

            rewards.append(current_rewards)

        data["model_" + str(num_training)] = np.array(rewards)

    np.savez(args.config + "/data", **data)
