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


def get_build_order(model_structure):
    def iter(structure, path, sequence):
        for k in structure.keys():
            element = structure[k]
            if type(element) is dict:
                iter(element, f"{path}/{k}", sequence)
            elif type(element) is Module and issubclass(element.module_cls, Network):
                sequence.insert(0, {'path': f"{path}/{k}", 'module': element})

    build_order = []
    iter(model_structure, '', build_order)
    print(build_order)
    return build_order



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
        #print(model_seq)
        self.seq = model_seq
        self.elements = {}
        self.values = {}
        self.items = {}
        unsorted_elements = {}
        unsorted_paths = {}
        subnetworks = {}
        modules = {}
        #print(model_seq)
        pos = 0
        elements_order = [None] * len(model_seq.keys())
        for k in model_seq.keys():
            element = self.seq[k]['m']
            if type(element) is Module and issubclass(element.module_cls, Network):
                #element.args['level'] = level + 1
                #module =element.build(params={'level': level+1})
                subnetworks[k] = self.seq[k]#['m']
                elements_order[pos] = k
            else:
                #module= element.build()
                modules[k] = self.seq[k]#['m']
                elements_order[pos] = k

            pos = pos + 1
            #print(f'{level * 4 * " "}{module}')

        for k in subnetworks.keys():
            n = subnetworks[k]['m']
            n_built = n.build()
            self.items[k.replace('/', '.')[1:]] = n_built

            if 'p' not in subnetworks[k]:
                unsorted_elements[k] = {'m': n_built}
            else:
                unsorted_elements[k] = {'m': n_built, 'p': subnetworks[k]['p']}

            for kk in n_built.elements.keys():
                #print(f'{k}{kk}')
                setattr(self, f'{k}{kk}'.replace('/', '_')[1:], n_built.elements[kk]['m'])

        for k in modules.keys():
            m = modules[k]['m']
            m_built = m.build()
            self.items[k.replace('/', '.')[1:]] = m_built

            if 'p' not in modules[k]:
                unsorted_elements[k] = {'m': m_built}
            else:
                unsorted_elements[k] = {'m': m_built, 'p': modules[k]['p']}
            #print(k)
            setattr(self, k.replace('/', '_')[1:], unsorted_elements[k]['m'])

        #print(elements_order)
        for k in elements_order:
            e = unsorted_elements[k]['m']
            if isinstance(e, Network):
                for kk in e.elements.keys():
                    #print(f'{k}{kk}')
                    self.elements[f'{k}{kk}'] = e.elements[kk]
                    #setattr(self, f'{k}{kk}'.replace('/', '_')[1:], self.elements[f'{k}{kk}'])
            else:
                #print(k)
                self.elements[k] = unsorted_elements[k]
                #setattr(self, k.replace('/', '_')[1:], self.elements[k])

        #print(f'\t\t{self.elements}')

        self.to('cuda')
        '''
        for k in self.elements.keys():
            #print(k)
            setattr(self, k.replace('/', '.')[1:], self.elements[k])
        
        #print(self.__dict__)
        '''

    def __getitem__(self, item):
        return self.items[item]

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

        all_out_store = {'module': m.serialize(), 'network': {}}
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
        for k in self.elements.keys():
            if not self.elements[k]['m'] is None:
                #print(k)

                #print('*' * 10)
                #print(k)
                if 'p' in self.elements[k]:
                    vs =  []
                    subseq =  self.elements[k]['p']
                    debug_output = "before: ["
                    for subp in subseq:
                        vs.append(self.values[subp])
                        debug_output = debug_output + f"{self.values[subp].shape}, "
                    x = self.elements[k]['m'](*vs)
                    debug_output = debug_output + "]"
                    #print(debug_output)
                else:
                    #print(f"before: {x.shape}")
                    x = self.elements[k]['m'](x)

                if type(x) is dict:
                    x = x[list(x.keys())[-1]]

                self.values[k] = x
                #print(f"after: {x.shape}")
                #print('*' * 10)

        #print(self.values['/feature/flatten'].shape)


        return self.values






'''
class Network(nn.Module):

    def __init__(self, sequence=None, name=None):#, model_cls):
        super(Network, self).__init__()
        #print(10 * '*')
        #print('\narchitecture')
        #print(pretty(self.architecture()))
        model_seq, name = nodes_to_sequence(self.architecture()) if sequence is None else (sequence, name)
        #print('\nmodel_seq')
        #print(pretty(model_seq))
        self.args = {} if not hasattr(self, 'args') else self.args
        self.seq = model_seq
        self.values = {}
        self.elements = {}
        self.name = name
        self.debug_order = []

        for k in self.seq:
            #print(self.seq[k]['m'])
            element = self.seq[k]['m'].build()
            #print(element)
            self.elements[k] = self.seq[k]['m'].build()

            self.debug_order.append(k)
            #print(k)
            if isinstance(self.elements[k], Network):
                #print(f'\t{self.elements[k].elements.keys()}')
                self.debug_order += self.elements[k].debug_order

            setattr(self, k.replace('/', '_')[1:], self.elements[k])


        #for k in self.elements:
        #    setattr(self, k.replace('/', '_'), self.elements[k])

        self.to('cuda')

    def flatten(self):
        def iter(path, obj):
            pass

        flattened = {}



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

                if type(x) is dict:
                    x = x[list(x.keys())[-1]]

                self.values[k] = x
                #print(f"after: {x.shape}")
                #print('*' * 10)

        #print(self.values['/feature/flatten'].shape)


        return self.values
        
'''
