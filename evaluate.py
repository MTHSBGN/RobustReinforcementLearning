import gym

from robust_reinforcement_learning.agents.ppo_agent import PPOAgent


if __name__ == "__main__":
    env = gym.make("BipedalWalker-v2")
    agent = PPOAgent(env.observation_space, env.action_space)
    agent.load()

    for _ in range(10):
        obs = env.reset()
        done = False
        while not done:
            action = agent.get_actions([obs])[0]
            obs, reward, done, _ = env.step(action)
            env.render()

