from gym.wrappers import TimeLimit
from gym.envs.registration import EnvSpec

from environments.cartpole_gravity import CartPoleGravityEnv
from environments.cartpole_length import CartPoleLengthEnv

GRAVITY_DESCRIPTION = [9.81, 1.0]
LENGTH_DESCRIPTION = [0.1, 1.0]


def make_cartpole_gravity():
    env = CartPoleGravityEnv(gravity_description=GRAVITY_DESCRIPTION)
    env.spec = EnvSpec("CustomCartPole-v0", reward_threshold=195.0)
    return TimeLimit(env, max_episode_steps=200.0)


def make_cartpole_length():
    env = CartPoleLengthEnv(LENGTH_DESCRIPTION)
    env.spec = EnvSpec("CustomCartPole-v0", reward_threshold=195.0)
    return TimeLimit(env, max_episode_steps=200.0)


def get_env_creator(env_name):
    if env_name == "CartPoleGravity":
        return make_cartpole_gravity

    return make_cartpole_length
