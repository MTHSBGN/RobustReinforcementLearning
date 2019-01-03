from gym.wrappers import TimeLimit
from gym.envs.registration import EnvSpec
from vec_env import SubprocVecEnv

from environments.cartpole_gravity import CartPoleGravityEnv
from environments.cartpole_length import CartPoleLengthEnv

GRAVITY_VALUES = [7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]
LENGTH_DESCRIPTION = [0.1, 1.0]


def make_environment(name, num_envs):
    """
    Create instances of one of the a custom environments defined in the other files of this directory.

    Parameters
    ----------
    name: string
        The name of the environment
    num_envs: int
        The number of environments to create

    Returns
    -------
    List(TimeLimit)
        The environment used for evaluation
    SubprocVecEnv
        The environments used for training

    Raises
    ------
    ValueError
        If the name given does not exist
    """

    evaluators = []
    if name is "CartPoleGravity":
        evaluators.append(CartPoleGravityEnv(GRAVITY_VALUES))
        evaluators.append(CartPoleGravityEnv(GRAVITY_VALUES, False))

    if name is "CartPoleLength":
        evaluators.append(CartPoleLengthEnv(LENGTH_DESCRIPTION))
        evaluators.append(CartPoleLengthEnv(LENGTH_DESCRIPTION, False))

    if len(evaluators) == 0:
        raise ValueError("The name given does not correspond to an existing environment")

    for i, env in enumerate(evaluators):
        env.spec = EnvSpec("CustomCartPole-v0", reward_threshold=195.0)
        evaluators[i] = TimeLimit(env, max_episode_steps=200.0)

    return evaluators, SubprocVecEnv([lambda: evaluators[0] for _ in range(num_envs)])
