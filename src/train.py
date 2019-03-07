import numpy as np

from vec_env import SubprocVecEnv


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
        agent.save(directory + "/model_" + str(num_training + 1))
