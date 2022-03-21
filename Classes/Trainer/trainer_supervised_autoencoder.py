import torch
from Library.trainerbase import TrainerBase

class TrainerAutoencoder(TrainerBase):
    def __init__(self,  classificationLoss=None,reconstructionLoss=None,weight=0.4,
                        optimizer=None, scheduler=None, actor_list=None, model=None, loaders=None, epochs=50, experiment_fn=None):
        super().__init__(criterion=classificationLoss, optimizer=optimizer, scheduler=scheduler, actor_list=actor_list, model=model, loaders=loaders, epochs=epochs, experiment_fn = experiment_fn)
        self.reconstructionLoss= reconstructionLoss
        self.classificationLoss = classificationLoss
        self.wp = 1.0
        self.wr = weight

    def train_batch_loop(self, model, trainloader, epoch):
        train_loss = 0.0
        train_acc = 0.0
        scaler = torch.cuda.amp.GradScaler()
        batch = 0
        for data in trainloader:
            images = data['working_image']
            labels = data['label']
            batch += 1
            # if batch > 1:
            # torch.cuda.synchronize()
            # self.starter_batch.record()
            # torch.cuda.synchronize()

            # sys.stdout.write(f"\rbatch: {batch} / {len(trainloader)}")
            # sys.stdout.flush()

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                images = images.to('cuda')
                labels = labels.to('cuda')

                data = model(images)
                x_hat = data['/decode_feature/trim4']
                y_hat = data['/classifier/out']
                loss =  self.wp * self.classificationLoss(y_hat, labels) +\
                        self.wr * self.reconstructionLoss(x_hat, images)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                train_loss += loss.item()
                # self.set_value('loss', loss.item)
                # self.set_value('epoch', epoch)
                # self.set_value('batch', batch)

            # curr_time = 0.0
            # if batch > 1:
            # torch.cuda.synchronize()

            # self.ender_batch.record()

            # curr_time = self.starter_batch.elapsed_time(self.ender_batch)

            self.listener.add_values([
                (loss.item(), 'loss', ['batch']),
                (epoch, 'epoch', ['batch']),
                (batch, 'iteration', ['batch']),
                # (curr_time, 'duration', ['batch'])

            ])

            self.listener.call_signal('next_batch')
            train_acc += self.accuracy(y_hat, labels)

        return train_loss / len(trainloader), train_acc / len(trainloader)
        #return super().train_batch_loop(model, trainloader, epoch)

    def valid_batch_loop(self, model, validloader):
        #print("*" * 20)
        #print("dies ist eine validloop")
        #print("*" * 20)
        return super().valid_batch_loop(model, validloader)
