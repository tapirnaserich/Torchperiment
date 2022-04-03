from Library.graph import Network
from Library.modules import Module, Value
from Library.modell import Flatten
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimpleConvBlock(Network):
    def __init__(self, channels_in, channels_out, kernel_size=3, kernel_stride=1, pooling_stride=2):
        self.register('channels_in', channels_in)
        self.register('channels_out', channels_out)
        self.register('kernel_size', kernel_size)
        self.register('pooling_stride', pooling_stride)
        self.register('kernel_stride', kernel_stride)

        super().__init__()

    def architecture(self):
        architecture = {}
        architecture['conv'] = Module(nn.Conv2d, {'in_channels': self.channels_in,
                                                  'out_channels': self.channels_out,
                                                  'kernel_size': self.kernel_size,
                                                  'padding': (self.kernel_size-1)//2,
                                                  'stride': self.kernel_stride,
                                                  'bias': False})
        architecture['debug'] = Module(VariableNet, {'blackboard': {}, 'in_channels': self.channels_out})
        architecture['debug1'] = Module(VariableNet, {'blackboard': {}, 'in_channels': self.channels_out})

        architecture['bn'] = Module(nn.BatchNorm2d, {'num_features': self.channels_out})
        architecture['relu'] = Module(nn.ReLU, {'inplace': True})
        architecture['max_pool'] = Module(nn.MaxPool2d, {'kernel_size': 3,
                                                         'stride': self.pooling_stride,
                                                         'padding': 1})
        return architecture



class VariableNet(Network):
    def __init__(self, blackboard, in_channels):
        self.register('blackboard', blackboard)
        self.register('in_channels', in_channels)
        self.register('out_channels', 0)

        super().__init__()

    def architecture(self):
        return {'identity': Module(nn.Conv2d, {'in_channels': self.in_channels, 'out_channels': self.in_channels, 'kernel_size': 3, 'padding': 1})}


class DebugParaNet(Network):
    def __init__(self):
        super().__init__()

    def architecture(self):
        return {
            #'stem': Module(VariableNet, {'blackboard': {}, 'in_channels': 1} ),
            'stem': Module(nn.Conv2d, {'in_channels':1,'out_channels':1,'kernel_size':3,'padding':1}),
            'features': Module(SimpleConvBlock, {'channels_in': 1,
                                                 'channels_out': 32,
                                                 'kernel_size': 3,
                                                 'kernel_stride': 2}),
            'debug0': {
                'element0': Module(nn.ReLU),
                'debug1': {
                    'network1': Module(VariableNet, {'blackboard': {},'in_channels': Value(self, 'features.channels_out')})
                }

            }

        }



class SimpleParaNet(Network):
    def __init__(self, blackboard):
        self.register('blackboard', blackboard)
        super().__init__()

    def architecture(self):
        return {
            'stem': Module(SimpleConvBlock, {'channels_in': 1, 'channels_out': 32, 'kernel_size': 3, 'kernel_stride': 2}),
            'features': Module(VariableNet, {'blackboard': self.blackboard, 'in_channels': Value(self, 'stem.conv.out_channels')}),
            'classifier': {
                'pooling': Module(nn.AdaptiveAvgPool2d, {'output_size': (4,4)}),
                'identity': Module(VariableNet, {'blackboard': self.blackboard, 'in_channels': 32}),
                'flatten': Module(Flatten),

            }
        }