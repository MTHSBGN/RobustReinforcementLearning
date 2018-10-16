from lib.experiment import Experiment
from lib.agents.ac_agent import ActorCriticAgent

exp = Experiment(ActorCriticAgent, "CartPole-v0")
exp.run(1000)
