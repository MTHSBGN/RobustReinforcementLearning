import gym

from lib.agents import Agent


class Experiment:
    def __init__(self, agent_name, env_name):
        self.simulator = gym.make(env_name)
        self.agent: Agent = agent_name()
        self.rewards = []
        self.mean_reward = 10

    def run(self, num_episode):
        for episode in range(num_episode):
            obs = self.simulator.reset()
            done = False

            self.rewards = []

            while not done:
                # self.simulator.render()
                action = self.agent.select_action(obs)
                obs, reward, done, _ = self.simulator.step(action)
                self.rewards.append(reward)

            self.episode_summary(episode)
            self.mean_reward = self.mean_reward * \
                0.99 + sum(self.rewards) * 0.01

            self.agent.improve(self.rewards)

        self.simulator.close()

    def episode_summary(self, episode):
        print("#############################")
        print("Episode {}:".format(episode))
        print("Reward: {}".format(sum(self.rewards)))
        print("Average: {}".format(self.mean_reward))

    def evaluate(self):
        pass
