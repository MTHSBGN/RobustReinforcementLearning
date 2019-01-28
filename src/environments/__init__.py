from gym.wrappers import TimeLimit
from gym.envs.registration import EnvSpec
from vec_env import SubprocVecEnv

from environments.cartpole_custom import CartPoleCustomEnv


def make_training_environment(config):
    env = None
    if config["name"] == "CartPoleCustom":
        env = CartPoleCustomEnv(**config["parameters"], true_value=True)

    if env is None:
        raise ValueError("The given name does not correspond to an existing environment")

    env.spec = EnvSpec("CustomCartPole-v0", reward_threshold=195.0)
    env = TimeLimit(env, max_episode_steps=200.0)

    return SubprocVecEnv([lambda: env for _ in range(config["num_envs"])])
