import os
from os.path import dirname

test_dir = dirname(__file__)


def get_resource_dir(module_file: str):
    test_path = os.path.splitext(module_file)[0].split("VariantCalling/test")[1]
    return f"{test_dir}/resources/{test_path}"
