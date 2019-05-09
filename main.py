import gym
import numpy as np

from robust_reinforcement_learning.agents.ppo_agent import PPOAgent
from robust_reinforcement_learning.buffer import Buffer

if __name__ == "__main__":
    num_envs = 8
    env_name = "BipedalWalker-v2"
    env = gym.make(env_name)
    agent = PPOAgent(env.observation_space, env.action_space)
    buffer = Buffer(agent, env_name, num_envs, env.observation_space, env.action_space, num_step=256)

    timestep = 0
    max_timestep = 10000000
    num_update = 0
    mean_reward = 0

    while mean_reward < env.spec.reward_threshold and timestep < max_timestep:
        buffer.collect()
        agent.update(buffer)
        num_update += 1
        timestep += len(buffer) * num_envs

        if num_update % 4 == 0:
            mean_reward = np.mean(buffer.rewards)
            agent.save()
            print("{}/{}: {}".format(timestep, max_timestep, mean_reward))

    mean_reward = np.mean(buffer.rewards)
    agent.save()
    print("{}/{}: {}".format(timestep, max_timestep, mean_reward))