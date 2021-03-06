# Related literature

## Reinforcement Leaning

<table>
    <tr>
        <td>Title</td>
        <td>Authors</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/1707.06347">Proximal Policy Optimization Algorithms</a></td>
        <td>John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov</td>
    </tr>
    <tr>
        <td colspan="2">They propose a new objective function in policy gradient methods. The idea is to perfom
            multiple epochs of training on a loss while keeping the policy distribution close to the previous one. To
            do so, they take a pessimistic bound between the ratio of policies and a clipped version of this ratio.</td>
    </tr>
</table>

## Adversarial networks

<table>
    <tr>
        <td>Title</td>
        <td>Authors</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/1611.01046">Learning to Pivot with Adversarial Networks</a></td>
        <td>Gilles Louppe, Michael Kagan, Kyle Cranmer</td>
    </tr>
    <tr>
        <td colspan="2">They introduce theoretical results for a training procedure based on adversarial networks for
            enforcing a pivotal property. A pivot is a quantity whose distribution does not depend on the unknown values
            of the nuisance parameters that parametrize this family of data generation processes</td>
    </tr>
</table>

## Textbooks and courses

<table>
    <tr>
        <td>Title</td>
        <td>Authors</td>
    </tr>
    <tr>
        <td><a href="http://incompleteideas.net/book/the-book-2nd.html">Reinforcement Learning: An Introduction</a></td>
        <td>Richard Sutton and Andrew Barto</td>
    </tr>
    <tr>
        <td><a href="https://sites.google.com/view/deep-rl-bootcamp/lectures">Deep RL Bootcamp</a></td>
        <td>Berkeley CA</td>
    </tr>
</table>

## TODO

- [Robust Adversarial Reinforcement Learning](https://arxiv.org/abs/1703.02702) - Lerrel Pinto, James Davidson, Rahul Sukthankar, Abhinav Gupta
- [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) - Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, Pieter Abbeel
- [Learning Dexterous In-Hand Manipulation](https://arxiv.org/abs/1808.00177) - OpenAI
- [Benchmarking Deep Reinforcement Learning for Continuous Control](https://arxiv.org/abs/1604.06778) - Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel
