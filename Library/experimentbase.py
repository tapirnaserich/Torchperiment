import os

import torch
from Library.modules import Module, ModuleDict
from Library.utils import serialize_dict_to_disk, pretty, deserialize_dict_from_disk, deserialize_dill_from_disk
import dill

class Experiment():

    def __init__(self, params, module_graph,
                 experiment_name = 'Test',
                 experiment_path = 'Experiments/Test',
                 init=True,
                 track_lambdas={
                     'iter_per_epoch': lambda experiment: len(experiment.loaders.train_loader)
                 }):
        self.experiment_name = experiment_name
        self.experiment_path = experiment_path
        self.params = params
        params['experiment_fn'] = self.get_experiment
        self.module_graph = module_graph
        self.all_modules = self.module_graph.build(params)



        self.track_lambdas = track_lambdas



        for k in self.all_modules.keys():
            setattr(self, k, self.all_modules[k])

        for k in self.params.keys():
            p = self.params[k]
            if type(p) is Module:
                setattr(self, k, self.params[k].build())
            else:
                setattr(self, k, self.params[k])

        for k in track_lambdas.keys():
            l = track_lambdas[k]
            setattr(self, k, l(self))

    def get_experiment(self):
        return self

    def serialize(self):
        out_experiment = {}
        out_experiment['modules'] = self.module_graph.serialize()

        out_params = {}
        for k in self.params.keys():
            p = self.params[k]
            if type(p) is Module:
                out_params[k] = p.serialize()
            elif not hasattr(p, '__call__') :
                out_params[k] = p

        out_experiment['params'] = out_params


        #if not os.path.isdir(self.experiment_path):
        #    os.mkdir(self.experiment_path)
        #serialize_dict_to_disk(out_experiment, f'{self.experiment_path}/experiment.csv')


        out_settings = {}
        out_settings['experiment_name'] = self.experiment_name
        #out_settings['track_values'] = self.track_values
        out_settings['track_lambdas'] = {}
        out_settings['experiment_path'] = self.experiment_path

        for k in self.track_lambdas.keys():
            l = self.track_lambdas[k]
            l_serialized = dill.dumps(l)
            dill_path = f"{self.experiment_path}/lambda_{k}.pkl"
            out_settings['track_lambdas'][k] = dill_path

            dill_file = open(dill_path, "wb")
            dill_file.write(dill.dumps(l_serialized))
            dill_file.close()


        out_experiment['settings'] = out_settings

        if not os.path.isdir(self.experiment_path):
            os.mkdir(self.experiment_path)

        serialize_dict_to_disk(out_experiment, f'{self.experiment_path}/experiment.csv')

        return out_experiment

    @staticmethod
    def Deserialize(dir_path):
        path = f"{dir_path}/experiment.csv"
        all_data = deserialize_dict_from_disk(path)
        s = all_data['settings']
        track_lambdas = {}
        for k in s['track_lambdas'].keys():
            p = s['track_lambdas'][k]
            track_lambdas[k] = deserialize_dill_from_disk(p)

        module_graph_serialized = all_data['modules']
        module_graph = ModuleDict.Deserialize(module_graph_serialized)

        params = {}
        for k in all_data['params'].keys():
            data = all_data['params'][k]
            if type(data) is dict and 'module_type' in data.keys():
                t = data['module_type']
                cls = t.split("'")[1].split('.')[-1]
                if cls == 'Module':
                    params[k] = Module.Deserialize(data)
            else:
                params[k] = data

        return Experiment(params, module_graph,
                          experiment_name=s['experiment_name'],
                          experiment_path=dir_path,
                          track_lambdas=track_lambdas)




    def fit(self):
        self.trainer.fit()




    def debug(self):
        print(self.params)
        for k in self.params.keys():
            p = self.params[k]
            if type(p) is Module:
                print(p.serialize())
                print(p.build())

        input = torch.Tensor(1,1,256, 256).cuda()
        output = self.model(input)
        print(self.reconstructionLoss)
        print(f"/classifier/out: {output['/classifier/out'].shape}")

        for k in self.track_lambdas.keys():
            fn = self.track_lambdas[k]
            print(f"{k}\n{fn(self)}")


