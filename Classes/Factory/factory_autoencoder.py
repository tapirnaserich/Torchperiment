import sys

from Library.modules import ModuleList, Module

from Classes.Dataset.BaseDataset import BaseDataset
from Classes.Trainer.trainer_supervised_autoencoder import TrainerAutoencoder
from Classes.Listener.print_and_store_actorslist import PrintAndStoreActorsList
from Classes.Loaders.loaders import BaseDataLoader
from Classes.Optimizer.Adam import AdamOptimizer
from Classes.Modells.modells import BaseHierarchicHalfDepthAutoencoder, DebugModell

from torchvision.transforms import Compose, ToPILImage, Resize, RandomResizedCrop, ToTensor, Normalize, Grayscale


def get_factory():
    print('from factory')
    #print(globals())

    return {
        'dataset': Module(BaseDataset),
        'model': Module(BaseHierarchicHalfDepthAutoencoder),
        'transform': ModuleList(
                Compose, [
                    Module(ToPILImage),
                    Module(Resize, dependencies=['size','interpolation']),
                    Module(RandomResizedCrop, dependencies=['size', 'scale']),
                    Module(ToTensor),
                    Module(Normalize, dependencies=['mean', 'std']),
                    Module(Grayscale, dependencies=['num_output_channels'])
                ]),
        'trainer': Module(TrainerAutoencoder),
        'actor_list': Module(PrintAndStoreActorsList),
        'loaders':  Module(BaseDataLoader),
        'optimizer': Module(AdamOptimizer),
    }

    #return {'test': 'test'}