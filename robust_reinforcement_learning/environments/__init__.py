import gym
from gym.wrappers import TimeLimit

from robust_reinforcement_learning.environments.custom_bipedal_walker import CustomBipedalWalker
from robust_reinforcement_learning.environments.custom_bipedal_walker import CustomBipedalWalkerHardcore


def env_creator(env_config):
    hardcore = env_config.get("hardcore", False)

    env = CustomBipedalWalkerHardcore(env_config) if hardcore else CustomBipedalWalker(env_config)
    env.spec = gym.spec("BipedalWalkerHardcore-v2" if hardcore else "BipedalWalker-v2")

    return TimeLimit(env)
