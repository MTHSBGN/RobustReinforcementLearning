import numpy as np


def rollouts(agent, env, nuisance, n=1000):
    rewards = []
    actions = []

    for i in range(n):
        state = env.reset(nuisance)
        done = False
        cum_reward = 0
        while not done:
            action = agent.select_actions([state])[0]
            actions.append(action)
            state, reward, done, _ = env.step(action)
            cum_reward += reward

        rewards.append(cum_reward)

    rewards = np.array(rewards)
    actions = np.array(actions)

    _, num_actions = np.unique(actions, return_counts=True)

    return [np.mean(rewards), np.std(rewards)] + num_actions.tolist()
