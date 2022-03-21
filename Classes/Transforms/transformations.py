import torchvision.transforms as T
from PIL import Image
from Library.modules import ModuleList, Module
from torchvision.transforms import Compose, ToPILImage, Resize, RandomResizedCrop, ToTensor, Normalize, Grayscale


def get_transforms_with_randomresizedcrop_module():
    ml = ModuleList(
        Compose, [
            Module(ToPILImage),
            Module(Resize, dependencies=['size','interpolation']),
            Module(RandomResizedCrop, dependencies=['size', 'scale']),
            Module(ToTensor),
            Module(Normalize, dependencies=['mean', 'std']),
            Module(Grayscale, dependencies=['num_output_channels'])
        ]
    )
    return ml