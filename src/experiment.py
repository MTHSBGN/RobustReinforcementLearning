from agent import Agent
from environment import Environment


class Experiment:
    def __init__(self, agent: Agent, env: Environment):
        self.agent = agent
        self.env = env

    def run(self, num_episode):
        mean_reward = 10

        for episode in range(num_episode):
            obs = self.env.reset()
            total_reward = 0
            while not self.env.isdone():
                obs, reward = self.env.step(self.agent.select_action(obs))
                total_reward += reward
                self.agent.observe(reward)

            mean_reward = mean_reward * 0.99 + total_reward * 0.01
            self.agent.improve()

            print("Episode {}: {}".format(episode, mean_reward))
