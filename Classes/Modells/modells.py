import torch.nn as nn
from Library.modell import Flatten, batch_norm, Add, Reshape, Identity, Trim
from Library.modules import Module
from Library.graph import Network


class DebugModell(Network):
    def __init__(self, depth):
        self.register('depth', depth)
        super().__init__()


    def architecture(self):
        return {
            'stem': {
                'input': Module(nn.Conv2d,{'in_channels':1, 'out_channels': self.depth, 'kernel_size':3}),
                'output_size': Module(nn.Conv2d,{'in_channels': self.depth, 'out_channels':1, 'kernel_size':1})

            }
        }























class BaseHierarchicHalfDepthAutoencoder(Network):
    def architecture(self):
        return {
            'stem': {
                'input': Module(nn.Conv2d,{'in_channels':1, 'out_channels':4, 'kernel_size':3, 'stride':4, 'padding':1, 'bias':False}),
                'bn1': Module(nn.BatchNorm2d,{'num_features': 4}),
                'relu1': Module(nn.ReLU, {'inplace':True}),
            },
            'encode_layer1': {
                'conv': Module(nn.Conv2d,{'in_channels':4, 'out_channels':8, 'kernel_size':3, 'stride':2, 'padding':1, 'bias':False}),
                'bn': Module(nn.BatchNorm2d,{'num_features': 8}),
                'relu': Module(nn.ReLU,{'inplace':True}),
                'pool': Module(nn.MaxPool2d, {'kernel_size':2})
                },
            'encode_layer2': {
                'conv': Module(nn.Conv2d,{'in_channels':8, 'out_channels':16, 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}),
                'bn': Module(nn.BatchNorm2d,{'num_features':16}),
                'relu': Module(nn.ReLU,{'inplace':True}),
            },
            'feature': {
                'pool': Module(nn.AdaptiveAvgPool2d,((4,4),)),
                'relu': Module(nn.ReLU,{'inplace':True}),
                'depth_reduction': Module(nn.Conv2d,{'in_channels':16, 'out_channels':8, 'kernel_size':1, 'bias':False}),
                'bn': Module(nn.BatchNorm2d,{'num_features':8}),
                'relu': Module(nn.ReLU,{'inplace':True}),
                'flatten': Module(Flatten,{}),
            },
            'classifier': {
                'out': Module(nn.Linear, {'in_features':128, 'out_features':2})
            },
            'decode_feature': {
                'out': (Module(Reshape,(-1, 8,4,4)), ['feature/flatten']),
                'depth_projection': Module(nn.ConvTranspose2d, {'in_channels':8, 'out_channels':16, 'kernel_size':1, 'bias':False}),
                'bn1': Module(nn.BatchNorm2d,{'num_features':16}),
                'relu1': Module(nn.ReLU,{'inplace':True}),

                'trans_conv1':Module(nn.ConvTranspose2d, {'in_channels':16, 'out_channels':16, 'kernel_size':3, 'stride':5}),
                'bn2': Module(nn.BatchNorm2d,{'num_features':16}),
                'relu2': Module(nn.ReLU,{'inplace':True}),
                'trim1': Module(Trim,(16,)),

                'trans_conv2': Module(nn.ConvTranspose2d, {'in_channels':16, 'out_channels':8, 'kernel_size':3, 'stride':2}),
                'bn3': Module(nn.BatchNorm2d,{'num_features':8}),
                'relu3': Module(nn.ReLU,{'inplace':True}),
                'trim2': Module(Trim,(32,)),

                'trans_conv3': Module(nn.ConvTranspose2d, {'in_channels':8, 'out_channels':4, 'kernel_size':3, 'stride':2}),
                'bn4': Module(nn.BatchNorm2d,{'num_features':4}),
                'relu4': Module(nn.ReLU,{'inplace':True}),
                'trim3': Module(Trim,(64,)),

                'trans_conv4':Module(nn.ConvTranspose2d, {'in_channels':4, 'out_channels':1, 'kernel_size':5, 'stride':4}),
                'sigmoid5': Module(nn.Sigmoid,{}),

                'trim4': Module(Trim,(256,)),
            }
        }




