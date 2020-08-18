import os
import sys
from subprocess import call


def dir_path(string, check_cloud_path=False):
    if os.path.isdir(string):
        return string
    if check_cloud_path:
        if not string.startswith('gs://') and not string.startswith('s3://'):
            raise ValueError(f'Invalid cloud path {string}\nMust be an s3 or gs path')
        return string
    else:
        raise NotADirectoryError(string)


def cloud_sync(cloud_path_in, local_dir_in, print_output=False, raise_error_is_file_exists=False):
    if not os.path.isdir(local_dir_in):
        raise NotADirectoryError(local_dir_in)
    dir_path(cloud_path_in, check_cloud_path=True)
    local_path = os.path.join(
        local_dir_in, 'cloud_sync', f'{cloud_path_in.split(":")[0]}', cloud_path_in.split("//")[1]
    )
    if os.path.isfile(local_path):
        if raise_error_is_file_exists:
            raise FileExistsError(f'target local file {local_path} exists')
        if print_output:
            sys.stdout.write(f'Local file {local_path} already exists, skipping...')
    else:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        is_gs = cloud_path_in.split(":")[0] == 'gs'
        cmd = f'{"gsutil" if is_gs else "aws s3"} cp {cloud_path_in} {local_path}'
        if print_output:
            sys.stdout.write(f'Downloading to {local_path}')
        call(cmd.split())
    return local_path
