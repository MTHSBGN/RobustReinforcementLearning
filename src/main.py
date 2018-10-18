from lib.experiment import Experiment
from lib.agents.ac_agent import ActorCriticAgent

exp = Experiment(
    ActorCriticAgent,
    "CartPole-v0",
    num_episodes=1000,
    batch_episode=10,
    summary=True,
    render=False
)

exp.run()
# exp.evaluate('test.pt')