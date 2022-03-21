from Library.Listener.listener_actor_base import ListenerActorBase
import sys

class ListenerActorPrinting(ListenerActorBase):

    def next_batch(self, all_values, experiment):
        batch = all_values['batch']['iteration'][-1]
        #duration = all_values['batch']['duration'][-1]
        sys.stdout.write(f"\rbatch: {batch} / {experiment().iter_per_epoch}")# (it took {duration} seconds)")
        sys.stdout.flush()

    def next_epoch(self, all_values, experiment):
        i = all_values['epoch']['iteration'][-1]
        avg_train_loss = all_values['epoch']['avg_train_loss'][-1]
        avg_train_acc = all_values['epoch']['avg_train_acc'][-1]
        avg_valid_loss = all_values['epoch']['avg_valid_loss'][-1]
        avg_valid_acc = all_values['epoch']['avg_valid_acc'][-1]
        duration = all_values['epoch']['duration'][-1]
        print("\n")
        print(f"Epoch : {i + 1} Train Loss : {avg_train_loss} Train Acc : {avg_train_acc}")
        print(f"Epoch : {i + 1} Valid Loss : {avg_valid_loss} Valid Acc : {avg_valid_acc}")
        print(f"Duration in seconds: {duration}")

    def valid_loss_decreased(self, all_values, experiment):
        valid_min_loss = all_values['epoch']['valid_min_loss'][-1]
        avg_valid_loss = all_values['epoch']['avg_valid_loss'][-1]
        print(f"\n")
        print(f"Valid_loss decreased {valid_min_loss} --> {avg_valid_loss}")

    def value_changed(self, all_values, experiment):
        pass