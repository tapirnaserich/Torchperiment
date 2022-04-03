from Library.graph import Network
from Library.modules import Module, Value
from Library.modell import Flatten
import torch.nn as nn

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
            architecture[f'layer{i}'] = Module(nn.Linear, {'in_features': self.in_features, 'out_features': self.out_features})
            architecture[f'relu{i}'] = Module(nn.ReLU)
            architecture[f'dropout{i}'] = Module(nn.Dropout, {'p': self.blackboard[f'layer_{i}_drop_out']})

            self.in_features = self.out_features

        return architecture


class ParentNet(Network):
    def __init__(self, blackboard, classes):
        self.register('blackboard', blackboard)
        self.register('classes', classes)
        super().__init__()

    def architecture(self):
        return {
            'input': Module(Flatten),
            'features': Module(VariableNet, {'blackboard':self.blackboard}),
            'classifier': {
                'out': Module(nn.Linear, {'in_features': Value(self, 'features.out_features'), 'out_features': self.classes}),
            }
            #'classifier': Module(nn.Linear, {'in_features': 128, 'out_features': self.classes})
        }