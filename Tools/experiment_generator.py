import os
import sys
from pathlib import Path
path = Path(os.path.dirname(__file__)).absolute().parent.as_posix()
sys.path.append(path)


import argparse
from jinja2 import Template
from Library.Generators.experiment_generation import experiment_generator
from Library.utils import compile_template



def generate_experiment():
    template_path = f'{path}/Library/Templates/experiment_generate_template.jinja2'



    args = args_parser.parse_args()
    print(path)
    experiment_path = f'{path}/{args.base_target}/{args.target}'
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    module = experiment_generator(base_path=args.base_path, path=args.path)
    setattr(args, 'params', module.dependencies)

    template = compile_template(template_path=template_path,
                                data=args)

    with open(f'{experiment_path}/settings.py', "w") as fh:
        fh.write(template)




if __name__ == '__main__':

    template_path =f'{path}/Library/Templates/experiment_generate_template.jinja2'

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('-bp', '--base_path',
                             action='store', type=str,
                             default='Classes.Factory')
    args_parser.add_argument('-p', '--path',
                             action='store', type=str,
                             default='factory_mnist')

    args_parser.add_argument('-bt', '--base_target',
                                  action='store', type=str,
                                  default='Experiments')
    args_parser.add_argument('-t', '--target',
                                action='store', type=str,
                                default='test')

    args = args_parser.parse_args()
    print(path)
    experiment_path = f'{path}/{args.base_target}/{args.target}'
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    module = experiment_generator(base_path=args.base_path, path=args.path)
    setattr(args, 'params', module.dependencies)

    template = compile_template(template_path=template_path,
                                data= args)

    with open(f'{experiment_path}/settings.py', "w") as fh:
        fh.write(template)

    #print(sys.path)
