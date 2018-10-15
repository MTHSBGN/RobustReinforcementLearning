import gym


class Environment:
    def __init__(self, name):
        self.simulator = gym.make(name)
        self.done = False

    def step(self, action):
        obs, reward, done, _ = self.simulator.step(action)
        self.done = done
        return obs, reward

    def isdone(self):
        return self.done

    def reset(self):
        self.done = False
        return self.simulator.reset()
