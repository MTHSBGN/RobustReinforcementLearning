from lib.experiment import Experiment
from lib.agents.ac_agent import ActorCriticAgent

exp = Experiment(
    ActorCriticAgent,
    "CartPole-v0",
    num_episodes=2,
    batch_episode=2,
    summary=False,
    render=False
)

exp.run()
