class BaseListener:
    def __init__(self, values_changed_fn=None, experiment=None):
        self.all_values = {}
        self.all_signals = {}
        self.values_changed_fn = values_changed_fn
        self.experiment = experiment

    def get_values_at(self, path=[]):
        iter = self.all_values
        for p in path:
            if not p in iter.keys():
                return (False, None)
            iter = iter[p]

        return (True, iter)

    def add_signal(self, name, fn):
        self.all_signals[name] = fn

    def call_signal(self, name):
        if not name in self.all_signals.keys():
            return False

        if self.all_signals[name] is None:
            return False

        self.all_signals[name](self.all_values, self.experiment)
        return True

    def add_topic(self, name, path=[]):
        iterator = self.all_values
        for p in path:
            iterator[p] = {} if not p in iterator.keys() else iterator[p]
            iterator = iterator[p]

        iterator[name] = []
        return True

    def add_topics(self, topics, path=[]):
        for t in topics:
            self.add_topic(t, path)

    def add_values(self, values):
        for v in values:
            self.add_value(*v)

    def add_value(self, value, topic, path=[]):
        ok, vals = self.get_values_at(path)
        if not ok:
            return False
        if not topic in vals.keys():
            return False

        vals[topic].append(value)

        if not self.values_changed_fn is None:
            self.values_changed_fn(self.all_values, topic, value, path)