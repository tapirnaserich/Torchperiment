from Library.Listener.listener_actor_base import ListenerActorBase
import sys
import pandas as pd
import torch
import os
from Library.utils import pretty, serialize_dict_to_disk
import pickle


class ListenerActorStoring(ListenerActorBase):


    def next_epoch(self, all_values, experiment):
        def to_filename(target_dir, epoch):
            return f"{target_dir}/models/epoch{epoch}.pt"



        trial = all_values['trial']['trial'][-1]
        target_path = experiment().path
        target_dir = f'{target_path}/Trial{trial.number}'

        if not os.path.isdir(f'{target_dir}/models'):
            os.mkdir(f'{target_dir}/models')



        iterations = all_values['epoch']['iteration']
        epoch = iterations[-1]
        torch.save(experiment().model.state_dict(), to_filename(target_dir, epoch))
        filename_column = list(map(lambda x: to_filename(target_dir, x),iterations))
        df = pd.DataFrame(all_values['epoch'])
        df['model_path'] = filename_column
        df.to_csv(f"{target_dir}/epoch_data.csv")


    def next_batch(self, all_values, experiment):
        batch_data = all_values['batch']
        trial = all_values['trial']['trial'][-1]
        target_path = experiment().path
        target_dir = f'{target_path}/Trial{trial.number}'


        df = pd.DataFrame(batch_data)
        df.to_csv(f'{target_dir}/batch_data.csv')

    def valid_loss_decreased(self, all_values, experiment):
        pass

    def value_changed(self, all_values, experiment):
        pass

    def next_trial(self, all_values, experiment):
        target_path = experiment().path

        trial = all_values['trial']['trial'][-1]
        target_dir = f'{target_path}/Trial{trial.number}'
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        serialize_dict_to_disk(experiment().model.serialize(), f'{target_dir}/Model.json')
        serialize_dict_to_disk(trial.params, f'{target_dir}/params.json')

        with open(f'{target_dir}/study.pkl', 'wb') as f:
            pickle.dump(experiment().study, f)
        #print(trial.params)

        pass


