import os

import torch
from Library.modules import Module, ModuleDict, ModuleList
from Library.utils import serialize_dict_to_disk, pretty, deserialize_dict_from_disk, deserialize_dill_from_disk
import dill
import optuna
from optuna.trial import TrialState
from Library.OptimizerModules.modules import SuggestionCategorical, Suggestion, Multiply, Blackboard
from copy import  deepcopy
import pickle
import pathlib


class OptimizationExperiment:
    def __init__(self, params, module_graph,
                 track_lambdas={
                     'iter_per_epoch': lambda experiment: len(experiment.loaders.train_loader)
                 }):
        self.params = params
        self.path = pathlib.Path().resolve()


        params['experiment_fn'] = self.get_experiment
        self.module_graph = module_graph
        self.track_lambdas = track_lambdas



        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self.objective, n_trials=100)

        pruned_trials = self.study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(self.study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = self.study.best_trial
        with open('study.pkl', 'wb') as f:
            pickle.dump(self.study, f)
        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))




    def objective(self, trial):

        working_params = {}
        for k in self.params.keys():

            p = self.params[k]

            if hasattr(p, 'module_cls') and (p.module_cls is Suggestion or p.module_cls is Blackboard or p.module_cls is Multiply or p.module_cls is SuggestionCategorical):
                m = p.build(params=self.params)
                m.sample(trial=trial)

                working_params[k] = m.value
            elif hasattr(p, 'module_cls'):
                working_params[k] = p.build(params=self.params)

            else:
                working_params[k] = p



        self.all_modules = self.module_graph.build(params=working_params)
        model = self.all_modules['model']
        #input = torch.Tensor(1, 1, 28, 28).cuda()
        #output = model(input)
        #print(output['/classifier/out'].shape)

        trainer = self.all_modules['trainer']

        for k in self.all_modules.keys():
            setattr(self, k, self.all_modules[k])

        for k in self.params.keys():
            p = self.params[k]
            if type(p) is Module:
                setattr(self, k, self.params[k].build())
            else:
                setattr(self, k, self.params[k])

        for k in self.track_lambdas.keys():
            l = self.track_lambdas[k]
            setattr(self, k, l(self))


        value = trainer.fit(trial)


        return value




    def get_experiment(self):
        return self
