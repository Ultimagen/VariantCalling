import os
from os.path import dirname

test_dir = dirname(__file__)


def get_resource_dir(module_name: str):
    test_path = module_name.replace('.', '/').replace('test/', '')
    return f'{test_dir}/resources/{test_path}'


def make_test_outputs_dir(module_name: str) -> str:
    test_path = module_name.replace('.', '/').replace('test/', '')
    test_outputs_dir = f'{test_dir}/test_outputs/{test_path}'
    os.makedirs(test_outputs_dir, exist_ok=True)
    return test_outputs_dir

