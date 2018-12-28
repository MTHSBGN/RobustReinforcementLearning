from agents import ActorCriticAgent
from experiment import Experiment

agent = ActorCriticAgent(5, 2, 256)
experiment = Experiment(agent, "CartPoleLength", 8)
experiment.run()
