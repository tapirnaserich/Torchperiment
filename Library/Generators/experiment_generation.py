import sys

from Library.modules import ModuleDict
import importlib


def experiment_generator(base_path, path):
    module_path = f'{base_path}.{path}'
    module = importlib.import_module(module_path)
    return ModuleDict(module.get_factory())

    #exec(open(r"D:\PythonProjects\BachelorArbeit\Torchperiment\Classes\Factory\factory_autoencoder.py").read())
    #print(sys.path)
    #m = Module(dict, {})
    #d = m.build()
    #MODULE_PATH = r"D:\PythonProjects\BachelorArbeit\Torchperiment\Classes\Factory\factory_autoencoder.py"
    #test_spec = importlib.util.spec_from_file_location('factory_autoencoder', MODULE_PATH)
    #test_module = importlib.util.module_from_spec(test_spec)
    #print(module.__dict__)
    #test_spec.loader.exec_module(test_module)
    #print(test_module.get_factory())
    #print(test_spec.__dict__)

    '''
    print(globals())
    print(get_factory())

    md = ModuleDict(locals()['get_factory']())
    print("kommt auch an!")
    #print(md)
    model = md.modules['model'].build()
    model_serialized = model.serialize()
    print(model)
    print(pretty(model_serialized))
    return model_serialized
    #model_deserialized, name = Network.Deserialize(model_serialized)
    #print(name)
    '''
