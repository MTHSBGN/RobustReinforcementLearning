from lib.experiment import Experiment
from lib.agents.ac_agent import ActorCriticAgent

exp = Experiment(
    ActorCriticAgent,
    "CartPole-v0",
    num_episodes=100000,
    batch_episode=1,
    summary=True,
    render=False
)

exp.run()
