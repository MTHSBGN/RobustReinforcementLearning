import gym

from robust_reinforcement_learning.agent import Agent
from robust_reinforcement_learning.buffer import Buffer
from robust_reinforcement_learning.logger import Logger

from robust_reinforcement_learning.models import PPO
import robust_reinforcement_learning.environments

config = {
    "env_name": "CustomBipedalWalker-v0",
    "num_envs": 8,
    "num_steps": 2048,
    "max_timesteps": 3_000_000,
    "gamma": 0.99,
    "lambda": 0.95,
    "num_epochs": 10,
    "minibatch_size": 64,
    "epsilon": 0.2
}

if __name__ == "__main__":
    num_envs = config["num_envs"]
    env_name = config["env_name"]
    env = gym.make(env_name)

    logger = Logger()
    agent = Agent(env, PPO(env, config))
    buffer = Buffer(agent, env, config)

    timestep = 0
    while timestep < config["max_timesteps"]:
        metrics = buffer.collect()
        logger.update(metrics)
        timestep += metrics["timesteps"]

        metrics = agent.train(buffer)
        logger.update(metrics)

        logger.plot()
