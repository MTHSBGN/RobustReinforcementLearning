from agents import ActorCriticAgent
from environments import make_environment
from experiment import evaluate, train, save

env_name = "CartPoleLength"
num_envs = 8

agent = ActorCriticAgent(5, 2, 256)
evaluators, envs = make_environment(env_name, num_envs)

train(agent, envs, 1000000)

# agent.load("ActorCriticAgent2019_01_17_10_31_13")

for env in evaluators:
    print(evaluate(agent, env))

save(agent, evaluators)

for env in evaluators:
    env.close()
