from lib.experiment import Experiment
from lib.agents.ac_agent import ActorCriticAgent

exp = Experiment(
    ActorCriticAgent,
    "CartPole-v0",
    num_episodes=1000,
    summary=True,
    render=False
)

exp.run()
