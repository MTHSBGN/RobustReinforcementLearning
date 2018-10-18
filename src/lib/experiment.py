import gym

from lib.agents import Agent


class Experiment:
    def __init__(self, agent_name, env_name, num_episodes=1000, batch_episode=10, summary=False, render=False):
        self.simulator = gym.make(env_name)
        self.agent: Agent = agent_name(
            self.simulator.action_space,
            self.simulator.observation_space
        )
        self.num_episodes = num_episodes
        self.batch_episode = batch_episode
        self.summary = summary
        self.render = render

        self.rewards = []
        self.mean_reward = 10
        self.diagnostic = {
            'current_episode': 0,
            'current_timestep': 0
        }

        self.stored_data = {
            'observations': [],
            'rewards': []
        }

    def run(self):
        for episode in range(self.num_episodes):
            obs = self.simulator.reset()
            done = False

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

            self.diagnostic['current_episode'] += 1
            if self.diagnostic['current_episode'] % self.batch_episode == 0:
                self.agent.improve(self.rewards)
                self.rewards = []

        self.simulator.close()

    def episode_summary(self, episode):
        print("#############################")
        print("Episode {}:".format(episode))
        print("Reward: {}".format(sum(self.rewards)))
        print("Average: {}".format(self.mean_reward))

    def evaluate(self):
        pass
