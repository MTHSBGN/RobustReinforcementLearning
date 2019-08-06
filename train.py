from datetime import datetime
import json
from pathlib import Path

from robust_reinforcement_learning.agent import Agent
from robust_reinforcement_learning.buffer import Buffer
from robust_reinforcement_learning.environments import env_creator
from robust_reinforcement_learning.logger import Logger
from robust_reinforcement_learning.models import PPO

config = {
    "env_config": {
        "nuisance": {
            "low": 0.1,
            "high": 2.5
        }
    },
    "env_name": "CustomBipedalWalker-v0",
    "num_envs": 8,
    "num_steps": 4000,
    "max_timesteps": 5_000_000,
    "gamma": 0.99,
    "lambda": 0.95,
    "num_epochs": 10,
    "minibatch_size": 64,
    "epsilon": 0.2,
    "adversary": True
}

if __name__ == "__main__":
    num_envs = config["num_envs"]
    env_name = config["env_name"]
    env = env_creator(config["env_config"])

    name = datetime.now().strftime("%y%m%d_%H%M%S")
    logdir = Path(Path(__file__).parent, "logs", name).absolute()
    logdir.mkdir(parents=True, exist_ok=True)
    config["logdir"] = str(logdir)

    with Path(logdir, "config.json").open("w") as f:
        json.dump(config, f)

    logger = Logger(logdir)
    model = PPO(env.observation_space, env.action_space, config)
    agent = Agent(model, config)
    buffer = Buffer(agent, config)

    agent.save()

    timestep = 0
    while timestep < config["max_timesteps"]:
        metrics = buffer.collect()
        logger.update(metrics)
        timestep += metrics["timesteps"]

        metrics = agent.train(buffer)
        logger.update(metrics)

        logger.plot()

    agent.save()
