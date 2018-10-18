from lib.experiment import Experiment
from lib.agents.ppo_agent import PPOAgent

exp = Experiment(
    PPOAgent,
    "CartPole-v0",
    num_episodes=1000,
    batch_episode=10,
    summary=True,
    render=False
)

exp.run()
# exp.evaluate('test.pt')