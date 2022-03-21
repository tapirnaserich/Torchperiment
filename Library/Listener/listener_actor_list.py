from Library.Listener.listener_actor_base import ListenerActorBase
import sys

class ListenerActorList(ListenerActorBase):
    def __init__(self, actors):
        self.actors = actors

    def next_batch(self, all_values, experiment):
        for a in self.actors:
            a.next_batch(all_values, experiment)

    def next_epoch(self, all_values, experiment):
        for a in self.actors:
            a.next_epoch(all_values, experiment)

    def valid_loss_decreased(self, all_values,experiment):
        for a in self.actors:
            a.valid_loss_decreased(all_values, experiment)

    def value_changed(self, all_values, experiment):
        for a in self.actors:
            a.value_changed(all_values, experiment)
