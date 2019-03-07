def rollouts(agent, env, nuisance, n=1000):
    rewards = []
    for i in range(n):
        state = env.reset(nuisance)
        done = False
        cum_reward = 0
        while not done:
            action = agent.select_actions([state])[0]
            state, reward, done, _ = env.step(action)
            cum_reward += reward

        rewards.append(cum_reward)

    return rewards
