import os

import pandas
import torch
import numpy as np
import torch.nn.functional as F
import sys
from datetime import datetime
import time
import pandas as pd
from Library.Listener.listener import BaseListener



class TrainerBase():

    def __init__(self,  criterion=None,
                        optimizer=None,
                        scheduler=None,
                        actor_list=None,
                        model=None,
                        loaders=None,
                        epochs=50,
                        experiment_fn=None):
        self.scheduler = scheduler
        self.criterion = criterion
        self.optimizer = optimizer
        self.actor_list = actor_list
        self.experiment_fn = experiment_fn
        self.listener = self.init_listener()
        self.starter_batch = torch.cuda.Event(enable_timing=True)
        self.ender_batch = torch.cuda.Event(enable_timing=True)
        self.starter_epoch = torch.cuda.Event(enable_timing=True)
        self.ender_epoch = torch.cuda.Event(enable_timing=True)
        self.epochs = epochs
        self.loaders = loaders
        self.model = model
        '''
        self.next_epoch_fn = next_epoch_fn
        self.next_batch_fn = next_batch_fn
        self.valid_loss_decreased_fn = valid_loss_decreased_fn
        self.value_changed_fn = value_changed_fn
        self.listener = self.init_listener()
        '''

    def init_listener(self):
        l = BaseListener(experiment=self.experiment_fn)

        values_to_track_epoch=['avg_train_acc', 'avg_train_loss', 'avg_valid_acc', 'avg_valid_loss', 'valid_min_loss', 'iteration', 'duration']
        values_to_track_minibatch=['loss','epoch', 'iteration']#, 'duration']

        l.add_topics(values_to_track_epoch, ['epoch'])
        l.add_topics(values_to_track_minibatch, ['batch'])

        l.add_signal('next_epoch', self.actor_list.next_epoch)
        l.add_signal('next_batch', self.actor_list.next_batch)
        l.add_signal('valid_loss_decreased', self.actor_list.valid_loss_decreased)

        # l.add_value("value", "loss", ["batch"])
        # l.add_value("snd_value", "avg_train_acc", ["epoch"])
        return l

    def train_batch_loop(self, model, trainloader, epoch):

        train_loss = 0.0
        train_acc = 0.0
        scaler = torch.cuda.amp.GradScaler()
        batch = 0
        for data in trainloader:
            images = data['working_image']
            labels = data['label']

            batch += 1
            #if batch > 1:
                #torch.cuda.synchronize()
                #self.starter_batch.record()
                #torch.cuda.synchronize()

            #sys.stdout.write(f"\rbatch: {batch} / {len(trainloader)}")
            #sys.stdout.flush()


            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                images = images.to('cuda')
                labels = labels.to('cuda')

                logits = model(images)['/classifier/out']
                loss = self.criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                train_loss += loss.item()
                #self.set_value('loss', loss.item)
                #self.set_value('epoch', epoch)
                #self.set_value('batch', batch)

            #curr_time = 0.0
            #if batch > 1:
                #torch.cuda.synchronize()

                #self.ender_batch.record()

                #curr_time = self.starter_batch.elapsed_time(self.ender_batch)

            self.listener.add_values([
                (loss.item(), 'loss', ['batch']),
                (epoch, 'epoch', ['batch']),
                (batch, 'iteration', ['batch']),
                #(curr_time, 'duration', ['batch'])

            ])

            self.listener.call_signal('next_batch')
            train_acc += self.accuracy(logits, labels)

        return train_loss / len(trainloader), train_acc / len(trainloader)

    def valid_batch_loop(self, model, validloader):

        valid_loss = 0.0
        valid_acc = 0.0

        for data in validloader:
            images = data['working_image']
            labels = data['label']
            with torch.cuda.amp.autocast():
                images = images.to('cuda')
                labels = labels.to('cuda')

                logits = model(images)['/classifier/out']
                loss = self.criterion(logits, labels)

                valid_loss += loss.item()
                valid_acc += self.accuracy(logits, labels)

        return valid_loss / len(validloader), valid_acc / len(validloader)




    def fit(self):
        #if experiment_name == '':
        #    experiment_name = model.name

        #experiment_name = f'{experiment_name}_{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}'

        #print(f"fitfunction:  {experiment_name}")

        valid_min_loss = np.Inf
        #os.mkdir(f'{self.experiments_base_path}/{experiment_name}')

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

            '''
            path = f"{self.experiments_base_path}/{experiment_name}/epoch{i+1}.pt"
            torch.save(model.state_dict(), path)
            self.set_value("model_path", path)
            df_out = self.get_df()
            df_out.to_csv(f"{self.experiments_base_path}/{experiment_name}/data.csv")
            '''



    def accuracy(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        top_p, top_class = y_pred.topk(1, dim=1)
        equals = top_class == y_true.view(*top_class.shape)
        return torch.mean(equals.type(torch.FloatTensor))
