import envs.cartpole
import ray
from ray.tune.registry import register_env
from ray.rllib.agents import pg, ppo
from ray.tune.logger import pretty_print


def env_creator(env_config):
    return envs.cartpole.CartPoleEnv()


register_env("cartpole", env_creator)
ray.init()

config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
agent = ppo.PPOAgent(config=config, env="cartpole")

for i in range(100):
    result = agent.train()
    print(pretty_print(result))
