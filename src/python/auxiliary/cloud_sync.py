import os
import sys
from google.cloud import storage
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


def download_from_gs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def download_from_s3(bucket_name, object_name, destination_file_name):
    client = boto3.Session(profile_name="default").client("s3")
    client.download_file(bucket_name, object_name, destination_file_name)


def cloud_sync(
    cloud_path_in,
    local_dir_in="/data",
    print_output=False,
    force_download=False,
    raise_error_is_file_exists=False,
        dry_run=False,
):
    """ Download a file from the cloud to a respective local directory with the same name

    Parameters
    ----------
    cloud_path_in: str
        Path to a file on the cloud (gs / s3), can also be a local file in which case nothing will happen
    local_dir_in: str, optional
        Local directory to which files will be downloaded, under subdirectory "cloud_sync"
    print_output: bool, optional
        Print log messages
    force_download: bool, optional
        If True, download file again even if file with the target name exists (default False)
    raise_error_is_file_exists: bool, optional
        If True and local file already exists, raise ValueError (default False)
    dry_run: bool, optional
        If True, return local path without downloading (default False)
    Returns
        local_path: str
            Local path to which file was downloaded, if cloud_path_in was an existing local file it is returned as is
    -------

    """
    if os.path.isfile(cloud_path_in):
        return cloud_path_in
    if not os.path.isdir(local_dir_in):
        raise NotADirectoryError(local_dir_in)
    dir_path(cloud_path_in, check_cloud_path=True)
    cloud_service = cloud_path_in.split(":")[0]
    bucket = cloud_path_in.split("/")[2]
    blob = "/".join(cloud_path_in.split("/")[3:])
    local_path = os.path.join(local_dir_in, "cloud_sync", cloud_service, bucket, blob,)
    if dry_run:
        return local_path
    if not force_download and os.path.isfile(local_path):
        if raise_error_is_file_exists:
            raise FileExistsError(f"target local file {local_path} exists")
        if print_output:
            sys.stdout.write(f"Local file {local_path} already exists, skipping...")
    else:
        if print_output:
            sys.stdout.write(f"Downloading to {local_path}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if cloud_service == "gs":
            download_from_gs(bucket, blob, local_path)
        elif cloud_service == "s3":
            download_from_s3(bucket, blob, local_path)
        else:
            raise NotImplementedError()

    return local_path
