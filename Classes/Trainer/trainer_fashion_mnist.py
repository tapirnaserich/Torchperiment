import torch
from Library.trainerbase import TrainerBase
import time
import numpy as np
import optuna

class TrainerFashionMnist(TrainerBase):
    def __init__(self,  classificationLoss=None,
                        optimizer=None, scheduler=None, actor_list=None, model=None, loaders=None, epochs=50, experiment_fn=None):
        super().__init__(criterion=classificationLoss, optimizer=optimizer(), scheduler=scheduler, actor_list=actor_list, model=model, loaders=loaders, epochs=epochs, experiment_fn = experiment_fn)


    def fit(self, trial):
        #if experiment_name == '':
        #    experiment_name = model.name

        #experiment_name = f'{experiment_name}_{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}'

        #print(f"fitfunction:  {experiment_name}")

        valid_min_loss = np.Inf
        avg_train_loss = 0
        avg_train_acc = 0
        avg_valid_loss = 0
        avg_valid_acc = 0
        #os.mkdir(f'{self.experiments_base_path}/{experiment_name}')

        self.listener.add_values([
            (trial, 'trial', ['trial'])
        ])
        self.listener.call_signal('next_trial')
        self.model.to('cuda')
        for i in range(self.epochs):
            start = time.time()
            self.model.train()
            avg_train_loss, avg_train_acc = self.train_batch_loop(self.model, self.loaders.train_loader, i+1)
            self.model.eval()
            avg_valid_loss, avg_valid_acc = self.valid_batch_loop(self.model, self.loaders.val_loader)

            if avg_valid_loss <= valid_min_loss:
                #print(f"\nValid_loss decreased {valid_min_loss} --> {avg_valid_loss}")
                valid_min_loss = avg_valid_loss

            #print(f"Epoch : {i + 1} Train Loss : {avg_train_loss} Train Acc : {avg_train_acc}")
            #print(f"Epoch : {i + 1} Valid Loss : {avg_valid_loss} Valid Acc : {avg_valid_acc}")

            delta_start = time.time()-start
            self.listener.add_values([
                (avg_train_loss, 'avg_train_loss', ['epoch']),
                (float(avg_train_acc), 'avg_train_acc', ['epoch']),
                (avg_valid_loss, 'avg_valid_loss', ['epoch']),
                (float(avg_valid_acc), 'avg_valid_acc', ['epoch']),
                (valid_min_loss, 'valid_min_loss', ['epoch']),
                (i, 'iteration', ['epoch']),
                (delta_start, 'duration', ['epoch'])

            ])

            self.listener.call_signal('next_epoch')
            trial.report(avg_valid_acc, i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return avg_valid_acc