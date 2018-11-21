from gym.envs.registration import register

register(
    id='CartPole-v4',
    entry_point='envs.cartpole:CartPoleEnv',
)
