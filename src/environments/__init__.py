from vec_env import SubprocVecEnv

from environments.cartpole_mod import CartPoleCustomEnv


def make_training_environment(config):
    env = None
    if config["name"] == "CartPoleCustom":
        env = CartPoleCustomEnv()

    if env is None:
        raise ValueError("The given name does not correspond to an existing environment")

    return SubprocVecEnv([lambda: env for _ in range(config["num_envs"])])


# def make_evaluation_environments(config):
#     envs = []
#     if config["name"] == "CartPoleCustom":
#         envs.append(CartPoleCustomEnv(**config["parameters"], true_value=True))
#         envs.append(CartPoleCustomEnv(**config["parameters"], true_value=False))
#
#     if len(envs) == 0:
#         raise ValueError("The given name does not correspond to an existing environment")
#
#     return envs
