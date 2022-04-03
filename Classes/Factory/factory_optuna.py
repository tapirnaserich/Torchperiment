import sys

from Library.modules import ModuleList, Module

from Classes.Dataset.BaseDataset import BaseDataset
from Classes.Trainer.trainer_fashion_mnist import TrainerFashionMnist
from Classes.Listener.print_and_store_actorslist import PrintAndStoreActorsList
from Classes.Loaders.loaders import BaseDataLoader
from Classes.Optimizer.Adam import Optimizer
from Classes.Modells.fashion_mnist import ParentNet

from torchvision.transforms import Compose, ToPILImage, Resize, RandomResizedCrop, ToTensor, Normalize, Grayscale


def get_factory():
    #print(globals())

    return {
        'dataset': Module(BaseDataset),
        'model': Module(ParentNet),
        'trainer': Module(TrainerFashionMnist),
        'actor_list': Module(PrintAndStoreActorsList),
        'loaders':  Module(BaseDataLoader),
        'optimizer': Module(Optimizer),
    }

    #return {'test': 'test'