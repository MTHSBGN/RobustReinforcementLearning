import ray
from ray.tune.logger import pretty_print

from lib.agents.pg_agent import PGAgent


if __name__ == "__main__":
    ray.init()
    agent = PGAgent(env='CartPole-v0')

    for i in range(1000):
        result = agent.train()
        print(pretty_print(result))
