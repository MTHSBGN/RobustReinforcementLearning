import gym
import numpy as np

import time

from agents.ppo_agent import PPOAgent
from buffer import Buffer

if __name__ == "__main__":
    env_name = "BipedalWalker-v2"
    env = gym.make(env_name)
    agent = PPOAgent(env.observation_space, env.action_space)
    buffer = Buffer(agent, env_name, 4, env.observation_space, env.action_space, num_step=128)

    timestep = 0
    max_timestep = 1000000
    num_update = 0
    mean_reward = 0

    while mean_reward < env.spec.reward_threshold and timestep < max_timestep:
        buffer.collect()
        agent.update(buffer)
        num_update += 1
        timestep += len(buffer)

        if num_update % 10 == 0:
            mean_reward = np.mean(buffer.rewards)
            agent.save()
            print("{}/{}: {}".format(timestep, max_timestep, mean_reward))
