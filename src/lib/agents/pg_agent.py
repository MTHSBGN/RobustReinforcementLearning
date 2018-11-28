from ray.rllib.agents.agent import Agent, with_common_config
from ray.rllib.optimizers import SyncSamplesOptimizer

from lib.policy_graphs.pg_policy_graph import PGPolicyGraph


DEFAULT_CONFIG = with_common_config({
    "num_workers": 0,
    "lr": 0.0004
})


class PGAgent(Agent):
    _agent_name = "PGCustom"
    _default_config = DEFAULT_CONFIG
    _policy_graph = PGPolicyGraph

    @classmethod
    def default_resource_request(cls, config):
        cf = merge_dicts(cls._default_config, config)
        return Resources(cpu=1, gpu=0, extra_cpu=cf["num_workers"])

    def _init(self):
        self.local_evaluator = self.make_local_evaluator(
            self.env_creator,
            self._policy_graph
        )

        self.remote_evaluators = self.make_remote_evaluators(
            self.env_creator,
            self._policy_graph,
            self.config["num_workers"],
            {}

        )

        self.optimizer = SyncSamplesOptimizer(
            self.local_evaluator,
            self.remote_evaluators,
            self.config["optimizer"]
        )

    def _train(self):
        prev_steps = self.optimizer.num_steps_sampled
        self.optimizer.step()
        result = self.optimizer.collect_metrics()
        result.update(
            timesteps_this_iter=self.optimizer.num_steps_sampled - prev_steps
        )
        return result
