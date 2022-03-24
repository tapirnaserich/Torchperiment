class Suggestion:
    def __init__(self, name, t, start, end):
        self.type = t
        self.start = start
        self.end = end
        self.name = name

    def iterate(self, data, prefix):
        prefix = f'{prefix}_{self.name}'
        data[prefix] = self
        return prefix, self

    def sample(self):
        return self.end


class Blackboard:
    def __init__(self, elements):
        self.elements = elements
        self.all_values = {}

    def sample(self):
        prefix = ''
        data = {}
        for e in self.elements:
            if type(e) is Suggestion:
                prefix, sugg = e.iterate(data, prefix)
                data[prefix] = sugg
            elif type(e) is Multiply:
                data = {**data, **e.iterate(data, prefix)}

        return data


class Multiply:
    def __init__(self, variable, names, elements):
        self.variable = variable
        self.elements = elements
        self.names = names

    def iterate(self, data, prefix):
        self.variable.iterate(data, prefix)

        prefix = f'{prefix}_{self.names}' if prefix != '' else f'{self.names}'
        how_often = self.variable.sample()
        for i in range(how_often):
            for e in self.elements:
                sub_prefix = f'{prefix}_{i}'
                e.iterate(data, sub_prefix)

        return data