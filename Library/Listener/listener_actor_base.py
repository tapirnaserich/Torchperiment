import abc


class ListenerActorBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def next_epoch(self, all_values, experiment):
        pass

    @abc.abstractmethod
    def next_batch(self, all_values, experiment):
        pass

    @abc.abstractmethod
    def valid_loss_decreased(self, all_values, experiment):
        pass

    @abc.abstractmethod
    def value_changed(self, all_values, experiment):
        pass

    @abc.abstractmethod
    def next_trial(self, all_values, experiment):
        pass

