import torch
from Library.trainerbase import TrainerBase

class TrainerMnist(TrainerBase):
    def __init__(self,  classificationLoss=None,
                        optimizer=None, scheduler=None, actor_list=None, model=None, loaders=None, epochs=50, experiment_fn=None):
        super().__init__(criterion=classificationLoss, optimizer=optimizer, scheduler=scheduler, actor_list=actor_list, model=model, loaders=loaders, epochs=epochs, experiment_fn = experiment_fn)


