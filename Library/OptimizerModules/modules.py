import random

class SuggestionCategorical:
    def __init__(self, name, values):
        self.name = name
        self.value = None
        self.prefix = ''
        self.values = values

    def iterate(self, data, prefix, trial):
        self.prefix = f'{prefix}_{self.name}' if prefix != '' else f'{self.name}'
        self.sample(trial)
        data[self.prefix] = self
        return self.prefix, self

    def sample(self, trial):
        self.value = trial.suggest_categorical(self.name, self.values)
        return self.value


    def iterate_debug(self, data, prefix):
        self.prefix = f'{prefix}_{self.name}' if prefix != '' else f'{self.name}'
        self.sample_debug()
        data[self.prefix] = self
        return self.prefix, self

    def sample_debug(self):
        self.value = self.values[0]


class Suggestion:
    def __init__(self, name, t, start, end, log=False):
        self.type = t
        self.start = start
        self.end = end
        self.name = name
        self.value = 0
        self.prefix = ''
        self.log = log

    def iterate(self, data, prefix, trial):
        self.prefix = f'{prefix}_{self.name}' if prefix != '' else f'{self.name}'
        self.sample(trial)
        data[self.prefix] = self
        return self.prefix, self

    def sample(self, trial):
        name = self.prefix if self.prefix != '' else self.name

        if self.type is float:
            self.value = trial.suggest_float(name, self.start, self.end, log=self.log)
        elif self.type is int:
            self.value = trial.suggest_int(name, self.start, self.end)

        return self.value

    def iterate_debug(self, data, prefix):
        self.prefix = f'{prefix}_{self.name}' if prefix != '' else f'{self.name}'
        self.sample_debug()
        data[self.prefix] = self
        return self.prefix, self

    def sample_debug(self):
        self.value = self.end


class Blackboard:
    def __init__(self, elements):
        self.elements = elements
        self.value = {}

    def sample(self, trial):
        prefix = ''
        data = {}
        for e in self.elements:
            if type(e) is Suggestion:
                prefix, sugg = e.iterate(data, prefix, trial)
                data[prefix] = sugg.value
            elif type(e) is Multiply:
                new_data = e.iterate(data, prefix, trial)
                for k in new_data.keys():
                    data[k] = new_data[k].value



        self.value = data
        return data

    def sample_debug(self):
        prefix = ''
        data = {}
        for e in self.elements:
            if type(e) is Suggestion:
                prefix, sugg = e.iterate_debug(data, prefix)
                data[prefix] = sugg.value
            elif type(e) is Multiply:
                new_data = e.iterate_debug(data, prefix)
                for k in new_data.keys():
                    data[k] = new_data[k].value

        return data



class Multiply:
    def __init__(self, variable, names, elements):
        self.variable = variable
        self.elements = elements
        self.names = names

    def iterate(self, data, prefix, trial):
        self.variable.iterate(data, prefix, trial)

        prefix = f'{prefix}_{self.names}' if prefix != '' else f'{self.names}'
        self.variable.sample(trial)
        for i in range(self.variable.value):
            for e in self.elements:
                sub_prefix = f'{prefix}_{i}'
                if type(e) is Suggestion:
                    ne = Suggestion(name=e.name, t=e.type, start=e.start, end=e.end, log=e.log)
                ne.iterate(data, sub_prefix, trial)



        return data

    def iterate_debug(self, data, prefix):
        self.variable.iterate_debug(data, prefix)

        prefix = f'{prefix}_{self.names}' if prefix != '' else f'{self.names}'
        self.variable.sample_debug()
        for i in range(self.variable.value):
            for e in self.elements:
                sub_prefix = f'{prefix}_{i}'
                e.iterate_debug(data, sub_prefix)

        return data
