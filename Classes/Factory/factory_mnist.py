import sys

from Library.modules import ModuleList, Module

from Classes.Dataset.BaseDataset import BaseDataset
from Classes.Trainer.trainer_mnist import TrainerMnist
from Classes.Listener.print_and_store_actorslist import PrintAndStoreActorsList
from Classes.Loaders.loaders import BaseDataLoader
from Classes.Optimizer.Adam import AdamOptimizer
from Classes.Modells.mnist import MnistNet

from torchvision.transforms import Compose, ToPILImage, Resize, RandomResizedCrop, ToTensor, Normalize, Grayscale


def get_factory():
    #print(globals())

    return {
        'dataset': Module(BaseDataset),
        'model': Module(MnistNet),
        'transform': ModuleList(
                Compose, [
                    Module(ToTensor),
                    Module(Normalize, dependencies=['mean', 'std']),
                    Module(Grayscale, dependencies=['num_output_channels'])
                ]),
        'trainer': Module(TrainerMnist),
        'actor_list': Module(PrintAndStoreActorsList),
        'loaders':  Module(BaseDataLoader),
        #'optimizer': Module(AdamOptimizer),
    }

    #return {'test': 'test'}