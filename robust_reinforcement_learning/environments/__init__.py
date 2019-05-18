import gym

gym.envs.register(
    id="CustomBipedalWalker-v0",
    entry_point="robust_reinforcement_learning.environments.custom_bipedal_walker:BipedalWalker",
    max_episode_steps=1600,
    reward_threshold=300
)
