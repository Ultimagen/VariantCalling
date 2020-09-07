#!/usr/bin/python3
# NOTE - this script relies on boto3, which is not installed in the system python by default. The option are either:
# 1. Add this line to ~/.bashrc and use it to call cloud_sync:
#  alias cloud_sync="conda run -n genomics.py3 python /home/ubuntu/proj/VariantCalling/src/python/scripts/cloud_sync.py"
# 2. Use only with "python cloud_sync.py" from an env where boto3 is installed
# 3. Install boto3 in the system python3


import argparse
import os
import sys
from subprocess import call
import boto3


def dir_path(string, check_cloud_path=False):
    if os.path.isdir(string):
        return string
    if check_cloud_path:
        if not string.startswith("gs://") and not string.startswith("s3://"):
            raise ValueError(f"Invalid cloud path {string}\nMust be an s3 or gs path")
        return string
    else:
        raise NotADirectoryError(string)


def cloud_sync(
    cloud_path_in, local_dir_in, print_output=False, raise_error_is_file_exists=False
):
    if not os.path.isdir(local_dir_in):
        raise NotADirectoryError(local_dir_in)
    dir_path(cloud_path_in, check_cloud_path=True)
    local_path = os.path.join(
        local_dir_in,
        "cloud_sync",
        f'{cloud_path_in.split(":")[0]}',
        cloud_path_in.split("//")[1],
    )
    if os.path.isfile(local_path):
        if raise_error_is_file_exists:
            raise FileExistsError(f"target local file {local_path} exists")
        if print_output:
            sys.stdout.write(f"Local file {local_path} already exists, skipping...\n")
    else:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        is_gs = cloud_path_in.startswith("gs://")
        is_s3 = cloud_path_in.startswith("s3://")

        if print_output:
            sys.stdout.write(f"Downloading from {'s3' if is_s3 else 'gs'} to {local_path}\n")
        if is_gs:
            cmd = f'{"gsutil" if is_gs else "aws s3"} cp {cloud_path_in} {local_path}'
            call(cmd.split())
        elif is_s3:
            spl = cloud_path_in.split(os.path.sep)
            bucket = spl[2]
            object_name = os.path.sep.join(spl[3:])
            client = boto3.Session(profile_name="default").client("s3")
            client.download_file(bucket, object_name, local_path)
        if print_output:
            sys.stdout.write(f"Download finished")
    return local_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download google storage file to local path"
    )
    parser.add_argument(
        "cloud_path", type=str, help="full path to aws s3 / google storage file"
    )
    parser.add_argument(
        "--local_dir",
        type=dir_path,
        default="/data",
        help="local directory the files will sync to",
    )
    args = parser.parse_args()
    cloud_path = args.cloud_path
    local_dir = args.local_dir
    cloud_sync(cloud_path, local_dir, print_output=True)
