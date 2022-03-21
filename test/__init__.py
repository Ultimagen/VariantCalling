import os
from os.path import dirname

test_dir = dirname(__file__)


def get_resource_dir(module_file: str):
    test_path = os.path.splitext(module_file)[0].split('VariantCalling/test')[1]
    return f'{test_dir}/resources/{test_path}'


def make_test_outputs_dir(module_file: str) -> str:
    test_path = os.path.splitext(module_file)[0].split('VariantCalling/test')[1]
    test_outputs_dir = f'{test_dir}/test_outputs/{test_path}'
    os.makedirs(test_outputs_dir, exist_ok=True)
    return test_outputs_dir

