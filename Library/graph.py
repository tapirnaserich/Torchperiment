import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from Library.utils import pretty
import pandas as pd
from importlib import import_module
from collections import OrderedDict
import copy
from Library.modules import Module
import abc





def nodes_to_sequence(model_structure):

    def iter(d, l, p):
        for k in d:
           if type(d[k]) is dict:
               iter(d[k], f"{l}/{k}", p)
           elif type(d[k]) is tuple:
               module, path = d[k]
               '''
               print('*'*10)
               print(k)
               print(path)
               print(l)
               print('*'*10)
               '''
               fullp = []
               for subp in path:
                   fullp.append(f"/{subp}")
               p[f"{l}/{k}"] = {"m": module,
                                       "p": fullp}
           else:
                p[f"{l}/{k}"] = {"m": d[k]}

    sequence = {}
    #model = model_structure()
    name = model_structure.__class__.__name__
    iter(model_structure, '', sequence)
    return sequence, name


class Network(nn.Module):

    def __init__(self, sequence=None, name=None):#, model_cls):
        super(Network, self).__init__()
        model_seq, name = nodes_to_sequence(self.architecture()) if sequence is None else (sequence, name)
        self.args = {} if not hasattr(self, 'args') else self.args
        self.seq = model_seq
        self.values = {}
        self.elements = {}
        self.name = name

        for k in self.seq:
            self.elements[k] = self.seq[k]['m'].build()

        for k in self.elements:
            setattr(self, k.replace('/', '_'), self.elements[k])

        self.to('cuda')

    def architecture(self):
        return None

    @staticmethod
    def Deserialize(serialized_model, name='Test', mode='network'):
        if mode == 'network':
            serialized_model = serialized_model['network']
            sequence = {}
            for k in serialized_model.keys():
                serialized_element = serialized_model[k]
                deserialized_element = {}
                deserialized_element['m'] = Module.Deserialize(serialized_element['m'])
                if 'p' in serialized_element.keys():
                    deserialized_element['p'] = serialized_element['p']
                sequence[k] = deserialized_element

            return Network(sequence, name)

        else:
            return None

    def serialize(self):
        m = Module(type(self), args = self.args)

        all_out_store = {'module': m, 'network': {}}
        for k in self.seq:
            to_store = self.seq[k]
            out_store = {}
            out_store['m'] = to_store['m'].serialize()

            if 'p' in to_store.keys():
                out_store['p'] = to_store['p']

            all_out_store['network'][k] = out_store

        return all_out_store

    def get_part_model(self, from_layer, to_layer, name='Test'):
        new_seq = {}
        keys = list(self.seq)
        keys_to_use = keys[keys.index(from_layer):keys.index(to_layer) + 1]
        for k in keys_to_use:
            new_seq[k] = self.seq[k]

        return Network(new_seq, name)

    def load_weights(self, model_path):
        state_dict = torch.load(model_path)
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            attr_name, attr_field = key.rsplit('.')
            if not hasattr(self, attr_name):
                continue

            #print(key)
            new_state_dict[key] = value

        self.load_state_dict(new_state_dict)

    def register(self, dependency, value):
        if not hasattr(self, 'args'):
            setattr(self, 'args', {})

        self.args[dependency] = value
        setattr(self, dependency, value)
        #print(self.dependencies)


    def forward(self, x):
        for k in self.seq:
            if not self.seq[k]['m'] is None:


                #print('*' * 10)
                #print(k)
                if 'p' in self.seq[k]:
                    vs =  []
                    subseq =  self.seq[k]['p']
                    debug_output = "before: ["
                    for subp in subseq:
                        vs.append(self.values[subp])
                        debug_output = debug_output + f"{self.values[subp].shape}, "
                    x = self.elements[k](*vs)
                    debug_output = debug_output + "]"
                    #print(debug_output)
                else:
                    #print(f"before: {x.shape}")
                    x = self.elements[k](x)

                self.values[k] = x
                #print(f"after: {x.shape}")
                #print('*' * 10)

        #print(self.values['/feature/flatten'].shape)


        return self.values