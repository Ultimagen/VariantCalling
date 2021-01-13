import pysam
import subprocess
import os
from os.path import join as pjoin
from google.cloud import storage
import pandas as pd


def count_lines_in_vcf(vcf_file):
    with pysam.VariantFile(vcf_file) as f_out:
        n_lines = sum((1 for _ in f_out))
    return n_lines

def count_length_of_interval_list(interval_list_path):
    with pd.read_csv(interval_list_path, comment='@', sep = '\t') as f_interval_list:
        sum_linterval = sum(f_interval_list.iloc[:,2]-f_interval_list.iloc[:,1])
    return sum_linterval


def run_shell_command(cmd, print_output=True):
    if print_output:
        print(cmd)

    if "'" in cmd or '|' in cmd:
        out = subprocess.check_output(cmd, shell=True)
    else:
        out = subprocess.check_output(cmd.split())

    if print_output:
        print(out)


def get_or_download_file(input_file, dirname):
    if os.path.isfile(input_file):
        return input_file
    elif input_file.startswith('gs://'):
        splt = input_file[len('gs://'):].split(os.path.sep)
        local_name = pjoin(dirname, splt[-1])
        if not os.path.isfile(local_name):
            gs_bucket = splt[0]
            client = storage.Client()
            bucket = client.get_bucket(gs_bucket)
            blob = storage.Blob(os.path.sep.join(splt[1:]), bucket)
            blob.download_to_filename(local_name)
        return local_name
    else:
        raise NotImplementedError(f'Expected either a file on google cloud or a local file, got {input_file}')


def get_tagged_output_file(tag, input_file, ext_insertion_index=None, ):
    if ext_insertion_index is None:
        if input_file.endswith('.gz'):
            ext_insertion_index = -2
        elif input_file.endswith('.bam'):
            ext_insertion_index = -1
        else:
            raise ValueError(f'Could not determine where to insert the tag in {input_file}')
    output_file = f"{'.'.join(input_file.split('.')[:ext_insertion_index])}.{tag}.{'.'.join(input_file.split('.')[ext_insertion_index:])}"
    return output_file
