class Agent:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def select_action(self, obs):
        _, act, value = self.model(obs, None)
        return act, value

    def train(self, data):
        for _ in range(self.config["minibatch_size"]):
            pass

        return self.model.update(data)

    def save(self):
        self.model.save(self.config["logdir"])

    def restore(self, checkpoint):
        pass
