import gym
import numpy as np

import time

from agents.ppo_agent import PPOAgent
from evaluate import Evaluator
from buffer import Buffer

if __name__ == "__main__":
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    agent = PPOAgent(env.observation_space, env.action_space)
    buffer = Buffer(agent, env_name, 8, env.observation_space, env.action_space, num_step=64)
    evaluator = Evaluator(env_name, 8)

    timestep = 0
    max_timestep = 1000000
    num_update = 0
    mean_reward = 0

    while mean_reward < env.spec.reward_threshold and timestep < max_timestep:
        if num_update % 10 == 0:
            mean_reward = evaluator.evaluate(agent)
            print("{}/{}: {}".format(timestep, max_timestep, mean_reward))

        buffer.collect()
        agent.update(buffer)
        num_update += 1
        timestep += len(buffer)
