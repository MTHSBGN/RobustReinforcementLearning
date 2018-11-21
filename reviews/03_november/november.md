# Review of November (21/11/2018)

## TODOs

- Complete formalization of the problem
- Complete PPO algorithm
- Determine the learning algorithm robust to the nuisance parameters + architecture

## Complete formalization of the problem

See [main.pdf](../../report/main.pdf)

## Complete PPO algorithm

The python library Ray and particulary is section RLlib has been used to develop the PPO alogithm.
This library allows us to write experiment that can be deployed on mulit-CPU or GPU computers in order to improve the training speed without changing the code.

A custom environment has been developed based on the CartPole. The difference between the two is that the custom version modify its gravity after each rollout an the said gravity is also given to the training agent.

## Determine the learning algorithm robust to the nuisance parameters + architecture

Learning algorithm:
- Select a value for the nuisance parameters according to an uniform distribution
- Generate rollouts according to current policy
- Train 

## Questions

- MDP --> Sequence obligatory or not?
- Varying input size in NN
  - Fix rollout or not
- Image paper Learning to Pivot
