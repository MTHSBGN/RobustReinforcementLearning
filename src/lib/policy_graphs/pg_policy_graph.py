import torch.optim as optim

from ray.rllib.evaluation.torch_policy_graph import TorchPolicyGraph
from ray.rllib.evaluation.postprocessing import compute_advantages

from lib.models.pg_model import PGModel, PGLoss


class PGPolicyGraph(TorchPolicyGraph):
    def __init__(self, observation_space, action_space, config):
        model = PGModel(observation_space, action_space, {})
        TorchPolicyGraph.__init__(
            self,
            observation_space,
            action_space,
            model,
            PGLoss(model),
            ["obs", "actions", "advantages"]
        )

    def optimizer(self):
        return optim.RMSprop(self._model.parameters(), lr=1e-4)

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None):
        sample_batch = compute_advantages(
            sample_batch,
            0.0,
            0.99,
            use_gae=False
        )

        # TODO Limit sizo of trajectory

        return sample_batch
