import gym

from agent import Agent

NUM_EPISODE = 10000

env = gym.make('CartPole-v0')
agent = Agent()

running_reward = 10


for episode in range(NUM_EPISODE):
    obs = env.reset()

    total_reward = 0

    done = False
    while not done:
        # env.render()
        action = agent.select_action(obs)
        obs, reward, done, _ = env.step(action)

        total_reward += reward
        agent.observe(reward)

    agent.improve()

    running_reward = running_reward * 0.99 + total_reward * 0.01
    print("Episode {}: {}".format(episode, running_reward))


env.close()
