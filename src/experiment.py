import numpy as np

from agents import Agent


def train(agent, envs, num_timesteps=1000000):
    """
    Train a RL agent in the given environment

    Parameters
    ----------
    agent: Agent
        An implemantation of the Agent's interface
    envs: List(gym.Envs)
        A vectorized version of an gym.Env using the SubprocVecEnv class
    num_timesteps: int
        The number of timesteps to train the agent on

    Returns
    -------
    None
    """

    state = envs.reset()
    timestep = 0
    while timestep < num_timesteps:
        observations = []
        actions = []
        rewards = []
        masks = []

        for _ in range(10):
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
