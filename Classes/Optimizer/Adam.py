from torch.optim import Adam

class AdamOptimizer(Adam):
    def __init__(self, lr, model):
        self.model = model
        super().__init__(lr=lr,params=self.model.parameters())

