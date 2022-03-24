import torch.nn as nn
from Library.graph import Network
from Library.modules import Module
from Library.modell import Flatten




class MnistNet(Network):
    def __init__(self, num_layers):
        self.register('num_layers', num_layers)
        super().__init__()


    def architecture(self):
        return {
            'feature': {
                'input': Module(nn.Conv2d, {'in_channels': 1, 'out_channels': 32, 'kernel_size': 3, 'padding': 1}),
                'relu1': Module(nn.ReLU),
                'conv1': Module(nn.Conv2d, {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'padding': 1}),
                'pool1': Module(nn.MaxPool2d, {'kernel_size': 2}),
                'dropout1': Module(nn.Dropout2d, {'p': 0.25}),
                'flatten': Module(Flatten),
            },
            'classifier': {
                'fc1': Module(nn.Linear, {'in_features': 12544, 'out_features':128}),
                'relu1': Module(nn.ReLU),
                'dropout2': Module(nn.Dropout, {'p': 0.5}),
                'out': Module(nn.Linear, {'in_features': 128, 'out_features': 10}),
            }

        }

