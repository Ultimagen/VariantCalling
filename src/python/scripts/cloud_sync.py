#!/usr/local/bin/python3

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from auxiliary.cloud_sync import cloud_sync, dir_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download aws s3 or google storage file to a corresponding local path"
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
    if not os.path.isdir(local_dir):
        raise ValueError(f"local_dir {local_dir} does not exist")
    cloud_sync(cloud_path, local_dir, print_output=True)
