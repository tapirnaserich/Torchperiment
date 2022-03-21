import os
from importlib import import_module
import copy
import inspect
from Library.utils import pretty, serialize_dict_to_disk, deserialize_dict_from_disk



class ModuleDict():
    def __init__(self, modules):
        self.modules = modules
        self.dependencies = self.find_dependencies()

    def find_dependencies(self):
        dependencies = []
        for k in self.modules.keys():

            m = self.modules[k]
            if m.dependencies is None:
                continue

            dependencies += m.dependencies

        return list(set(dependencies) - set(self.modules.keys()))

    @staticmethod
    def Deserialize(serialized):
        graph = {}
        for k in serialized.keys():
            data = serialized[k]
            cls = data['module_type']
            m_type = cls.split("'")[1].split(".")[-1]
            if m_type == 'Module':
                graph[k] = Module.Deserialize(serialized[k])
            elif m_type == 'ModuleList':
                graph[k] = ModuleList.Deserialize(serialized[k])


        return ModuleDict(graph)

    def serialize(self):
        out_modules = {}
        for k in self.modules.keys():
            m = self.modules[k]
            t = type(m)
            if t is Module:
                out_modules[k] = Module(m.module_cls, dependencies=m.dependencies).serialize()
            elif t is ModuleList:
                out_modules[k] = ModuleList(m.module_cls, m.modules, dependencies=m.dependencies).serialize()

        return out_modules

    def build(self, params = None):
        unresolved_dependencies = list(self.modules.keys())

        params_to_use = {}
        for k in params.keys():
            p = params[k]

            if type(p) is Module or type(p) is ModuleList:
                p = params[k].build()

            params_to_use[k] = p


        stack_to_build = []
        build_modules = {}
        print(params_to_use)
        for k in self.modules.keys():
            if not k in unresolved_dependencies:
                continue

            stack_to_build.append(k)
            while len(stack_to_build) > 0:
                key = stack_to_build[-1]
                module = self.modules[key]
                deps = module.dependencies if module.dependencies is not None else []
                found_unresolved = []

                for d in deps:
                    if d in unresolved_dependencies:
                        found_unresolved.append(d)
                        stack_to_build.append(d)

                if len(found_unresolved) == 0:
                    stack_to_build.pop()
                    if key in unresolved_dependencies:
                        unresolved_dependencies.remove(key)

                    print(module)
                    print('*' * 30)
                    build_module = module.build(params_to_use)
                    params_to_use[key] = build_module
                    build_modules[key] = build_module

        return build_modules

    def __repr__(self):
        out = ""

        for k in self.modules.keys():
            out += f'{k}\n'
            out += f'\t{self.modules[k]}\n\n'


        return out

class ModuleList():
    def __init__(self, module_cls, modules = [], dependencies = None, args = None):
        self.module_cls = module_cls
        self.modules = modules
        self.dependencies = self.find_dependencies() if dependencies is None else dependencies
        self.args = args

    def find_dependencies(self):
        all_dependencies = []
        for m in self.modules:
            all_dependencies += m.dependencies if not m.dependencies is None else []

        all_dependencies = list(set(all_dependencies))
        return all_dependencies

    def serialize(self):
        cls_split = str(self.module_cls).split("'")
        out_name = cls_split[1] if len(cls_split) > 1 else cls_split[0]


        out_modules = []
        for m in self.modules:
            out_modules.append(m.serialize())

        return {'module_type': str(type(self)),
                'module_cls': out_name,
               'modules': out_modules,
               'dependencies': self.dependencies}

    @staticmethod
    def Deserialize(module_serialized):
        module_cls = module_serialized['module_cls']
        module_path, class_name = module_cls.rsplit('.', 1)
        module = import_module(module_path)
        module_cls = getattr(module, class_name)

        dependencies = module_serialized['dependencies']

        modules = []
        for m in module_serialized['modules']:
            modules.append(Module.Deserialize(m))

        return ModuleList(module_cls, modules, dependencies)


    def build(self, params = None):
        all_modules = []
        for m in self.modules:
            all_modules.append(m.build(params))

        return self.module_cls(all_modules)

    def __repr__(self):
        cls_split = str(self.module_cls).split("'")
        out_name = cls_split[1] if len(cls_split) > 1 else cls_split[0]
        out_name = out_name.split('.')[-1]
        out =  f"{out_name}\n"

        for m in self.modules:
            out = out + f"\t {m} \n"

        return out


class Module():
    def __init__(self, module_cls, args = None, dependencies = None):
        self.module_cls = module_cls
        self.args = args
        self.dependencies = dependencies if dependencies is not None else []
        self.signature = {}
        if len(self.dependencies) == 0:
            deps = {}

            if 'Classes' in str(self.module_cls):
                deps = inspect.signature(self.module_cls.__init__).parameters

            elif 'function' in str(self.module_cls):
                deps =inspect.signature(self.module_cls).parameters

            for d in deps.keys():
                if d == 'self':
                    continue
                self.dependencies.append(d)

    def build(self, params = None):
        #print(self.module_cls)
        #print(f"\t{self.args}")
        #print(f"\t{self.dependencies}")
        #print("\n")
        args = copy.deepcopy(self.args) if not self.args is None or type(self.args) is tuple else {}
        #print(self.module_cls)
        #print('*' * 30)

        if      not self.dependencies is None and \
                not params is None and \
                not type(args) is tuple:


            for d in self.dependencies:
                if params is not None and d in params.keys():
                    args[d] = params[d]
        #print(f"args: {args}")
        self.args = args

        if len(args) == 0:
            return self.module_cls()
        elif len(args) > 0 and type(args) is dict:
            return self.module_cls(**args)

        return self.module_cls(*self.args)


    def serialize(self):
        cls_split = str(self.module_cls).split("'")
        out_name = cls_split[1] if len(cls_split) > 1 else cls_split[0]
        out_args = self.args

        return {'module_type': str(type(self)),
                'module_cls': out_name,
                'args': out_args,
                'dependencies': self.dependencies}

    @staticmethod
    def Deserialize(module_serialized):
        module_cls = module_serialized['module_cls']
        args = module_serialized['args']
        dependencies = module_serialized['dependencies']
        module_path, class_name = module_cls.rsplit('.', 1)
        module = import_module(module_path)
        module_cls = getattr(module, class_name)
        return Module(module_cls, args, dependencies)

    def __repr__(self):
        cls_split = str(self.module_cls).split("'")
        out_name = cls_split[1] if len(cls_split) > 1 else cls_split[0]
        out_name = out_name.split('.')[-1]
        return f"{out_name} with {self.args} and {self.dependencies} as dependencies"


