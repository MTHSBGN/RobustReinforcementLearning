import gym

from lib.agents import Agent


class Experiment:
    def __init__(self, agent_name, env_name, num_episodes=1000, summary=False, render=False):
        self.simulator = gym.make(env_name)
        self.agent: Agent = agent_name()
        self.num_episodes = num_episodes
        self.summary = summary
        self.render = render

        self.rewards = []
        self.mean_reward = 10

    def run(self):
        for episode in range(self.num_episodes):
            obs = self.simulator.reset()
            done = False

            self.rewards = []

            while not done:
                if self.render:
                    self.simulator.render()

                action = self.agent.select_action(obs)
                obs, reward, done, _ = self.simulator.step(action)
                self.rewards.append(reward)

            if self.summary:
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
