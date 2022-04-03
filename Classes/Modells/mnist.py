import torch.nn as nn
from Library.graph import Network
from Library.modules import Module
from Library.modell import Flatten



class VariableNet(Network):
    def __init__(self, blackboard, in_features=28*28):
        self.register('blackboard', blackboard)
        self.register('in_features', in_features)
        self.out_features = 0
        super().__init__()

    def architecture(self):
        architecture = {}

        for i in range(self.blackboard['n_layers']):
            self.out_features = self.blackboard[f'layer_{i}_units']
            p_dropout = self.blackboard[f'layer_{i}_drop_out']
            architecture[f'layer{i}'] = Module(nn.Linear, {'in_features': self.in_features, 'out_features': self.out_features})
            architecture[f'relu{i}'] = Module(nn.ReLU)
            architecture[f'dropout{i}'] = Module(nn.Dropout, {'p': p_dropout})

            self.in_features = self.out_features

        return architecture



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

