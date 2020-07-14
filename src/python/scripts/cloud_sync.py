#!/usr/local/bin/python3

import os
from subprocess import call
import argparse
import sys


def dir_path(string, check_cloud_path=False):
    if os.path.isdir(string):
        return string
    if check_cloud_path:
        if not string.startswith('gs://') and not string.startswith('s3://'):
            raise ValueError(f'Invalid cloud path {string}\nMust be an s3 or gs path')
        return string
    else:
        raise NotADirectoryError(string)


def cloud_sync(cloud_path_in, local_dir_in, print_output=False):
    if not os.path.isdir(local_dir_in):
        raise NotADirectoryError(local_dir_in)
    dir_path(cloud_path_in, check_cloud_path=True)
    local_path = os.path.join(
        local_dir_in, 'cloud_sync', f'{cloud_path_in.split(":")[0]}', cloud_path_in.split("//")[1]
    )
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    is_gs = cloud_path_in.split(":")[0] == 'gs'
    cmd = f'{"gsutil" if is_gs else "aws s3"} cp {cloud_path} {local_path}'
    if print_output:
        sys.stdout.write(f'Downloading to {local_path}')
    call(cmd.split())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download google storage file to local path')
    parser.add_argument('cloud_path', type=str, help='full path to aws s3 / google storage file')
    parser.add_argument(
        '--local_dir', type=dir_path, default='/cache', help='local directory the files will sync to'
    )
    args = parser.parse_args()
    cloud_path = args.cloud_path
    local_dir = args.local_dir
    cloud_sync(cloud_path, local_dir, print_output=True)
