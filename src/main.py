from lib.experiment import Experiment
import lib.agents


exp = Experiment(
    lib.agents.ActorCriticAgent,
    "CartPole-v0",
    num_episodes=10,
    summary=False,
    render=True
)

exp.run()
