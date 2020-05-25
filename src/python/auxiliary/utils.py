import pysam
import subprocess
import os
from os.path import join as pjoin
from google.cloud import storage


def count_lines_in_vcf(vcf_file):
    with pysam.VariantFile(vcf_file) as f_out:
        n_lines = sum((1 for _ in f_out))
    return n_lines


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


def parse_output_file(tag, input_file=None, output_file=None, ):
    if output_file is None:
        if input_file is None:
            raise ValueError('input_file and output_file cannpt both be None')
        if not input_file.endswith('.vcf.gz'):
            raise ValueError(
                f'''Could not automatically determine output file name. Expected input to end with .vcf.gz, got 
{input_file}'''
            )
        output_file = f"{'.'.join(input_file.split('.')[:-2])}.{tag}.{'.'.join(input_file.split('.')[-2:])}"
    return output_file
