# Review of September (14/09/2018)

## TODOs

- Formalization of the problem
- Definition of a benchmark
- Related literature

## Formalization of the problem

See formalization.pdf

## Definition of a benchmark

As benchmark environments, [multiple continuous control tasks](http://gym.openai.com/envs/#mujoco) from [OpenAI Gym](https://gym.openai.com) and the [MuJoCo physics simulator](http://www.mujoco.org/) will be used.

The focus will be on those three but more may be added if needed:

- [HalfCheetah](http://gym.openai.com/envs/HalfCheetah-v2/)
- [Ant](http://gym.openai.com/envs/Ant-v2/)
- [Humanoid](http://gym.openai.com/envs/Humanoid-v2/)

The nuisance parameters have not been decided yet but here are some possible canditates:

- Gravity
- Friction
- Mass
- Externel forces (e.g. cross-wind)

## Related literature

- [Learning to Pivot with Adversarial Networks](https://arxiv.org/abs/1611.01046) - Gilles Louppe, Michael Kagan, Kyle Cranmer
- [Robust Adversarial Reinforcement Learning](https://arxiv.org/abs/1703.02702) - Lerrel Pinto, James Davidson, Rahul Sukthankar, Abhinav Gupta
- [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) - Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, Pieter Abbeel
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) - Richard Sutton and Andrew Barto
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
