from torch.optim import Adam
import torch.optim as optim

class AdamOptimizer(Adam):
    def __init__(self, lr, model):
        self.model = model
        super().__init__(lr=lr,params=self.model.parameters())




class Optimizer():
    def __init__(self, optimizer_name, lr, model):
        self.model = model
        self.name = optimizer_name
        self.lr = lr

        self.optimizer = getattr(optim, self.name)(self.model.parameters(), lr=self.lr)

    def __call__(self):
        return self.optimizer

