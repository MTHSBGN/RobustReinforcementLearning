class Agent:
    def __init__(self, obs_space, act_space):
        self.obs_space = obs_space
        self.act_space = act_space

    def get_actions(self, observations, extra_returns=None):
        raise NotImplementedError

    def update(self, data):
        raise NotImplementedError
