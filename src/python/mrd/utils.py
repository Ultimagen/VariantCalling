import os
from os.path import join as pjoin, basename, dirname
import re
import subprocess


my_env = os.environ.copy()
Popen_args = dict(env=my_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def get_shard_paths(base_path):
    shards, stderr = subprocess.Popen(
        f"gsutil ls {base_path}".split(), **Popen_args
    ).communicate()
    if len(shards) == 0:
        raise ValueError(
            f"""Invalid HaplotypeCallerResultsBasePath
Command "gsutil ls {base_path}" returned with error:
{stderr.decode()}"""
        )
    shards = shards.decode().split(os.linesep)
    shards = [x for x in shards if len(x) > 0]  # remove blank line
    return shards


def process_shard(shard):
    try:
        shard_num = -1
        shard_num = int(re.match("gs://.*shard-([0-9]+)/.*", shard)[1])

        bucket_contents = (
            subprocess.check_output(f"gsutil ls {shard}".split())
            .decode()
            .strip()
            .split(os.linesep)
        )
        f_haps = None
        f_vcf = None
        f_vcf_ind = None
        max_attempt = None
        path_max_attempt = None

        for f in bucket_contents:
            if basename(f) == "haps.bam":
                f_haps = f
            elif f.endswith(".vcf.gz"):
                f_vcf = f
            elif f.endswith(".vcf.gz.tbi"):
                f_vcf_ind = f
            elif basename(f).strip() == "" and basename(dirname(f)).startswith(
                "attempt"
            ):
                n_attempt = int(basename(dirname(f)).split("-")[1])
                if max_attempt is None or n_attempt > max_attempt:
                    max_attempt = n_attempt
                    path_max_attempt = f

        if (f_haps is None) or (f_vcf is None) or (f_vcf_ind is None):
            if max_attempt is not None:
                return process_shard(path_max_attempt)
            else:
                return -1, shard_num, dict()
        return 0, shard_num, dict(f_haps=f_haps, f_vcf=f_vcf, f_vcf_ind=f_vcf_ind)
    except ValueError:
        return -1, shard_num, dict()
    except TypeError:
        return -1, shard_num, dict()
