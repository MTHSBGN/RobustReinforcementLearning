import gym

from agent import Agent
from environment import Environment
from experiment import Experiment

env = Environment('CartPole-v0')
agent = Agent()

exp = Experiment(agent, env)
exp.run(1000)

