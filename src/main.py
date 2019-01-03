from agents import ActorCriticAgent
from environments import make_environment
from experiment import evaluate, train

env_name = "CartPoleGravity"
num_envs = 8

agent = ActorCriticAgent(5, 2, 256)
evaluators, envs = make_environment(env_name, num_envs)

# train(agent, envs, 1000000)

for env in evaluators:
    print(evaluate(agent, env))

