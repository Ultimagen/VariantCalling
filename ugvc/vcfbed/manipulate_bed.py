from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings

import pybedtools
from simppl.simple_pipeline import SimplePipeline

from ugvc import logger
from ugvc.utils.exec_utils import print_and_execute

warnings.filterwarnings("ignore")

bedtools = "bedtools"


def filter_by_bed_file(in_bed_file, filtration_cutoff, filtering_bed_file, prefix, tag):
    out_filename = os.path.basename(in_bed_file)
    out_filtered_bed_file = prefix + out_filename.rstrip(".bed") + "." + tag + ".bed"
    cmd = (
        bedtools
        + " subtract -N -f "
        + str(filtration_cutoff)
        + " -a "
        + in_bed_file
        + " -b "
        + filtering_bed_file
        + " > "
        + out_filtered_bed_file
    )

    os.system(cmd)
    filtered_out_records = prefix + out_filename.rstrip(".bed") + "." + tag + ".filtered_out.bed"
    cmd = (
        bedtools
        + " subtract -N -f "
        + str(filtration_cutoff)
        + " -a "
        + in_bed_file
        + " -b "
        + out_filtered_bed_file
        + " > "
        + filtered_out_records
    )

    os.system(cmd)
    out_annotate_file = prefix + out_filename.rstrip(".bed") + "." + tag + ".annotate.bed"
    cmd = "cat " + filtered_out_records + ' | awk \'{print $1"\t"$2"\t"$3"\t"$4"|' + tag + "\"}' > " + out_annotate_file
    os.system(cmd)

    return out_annotate_file


def filter_by_length(bed_file, length_cutoff, prefix):
    out_filename = os.path.basename(bed_file)
    out_len_file = prefix + out_filename.rstrip(".bed") + ".len.bed"
    cmd = "awk '$3-$2<" + str(length_cutoff) + "' " + bed_file + " > " + out_len_file
    os.system(cmd)
    out_len_annotate_file = prefix + out_filename.rstrip(".bed") + ".len.annotate.bed"
    cmd = "cat " + out_len_file + ' | awk \'{print $1"\t"$2"\t"$3"\t"$4"|LEN"}\' > ' + out_len_annotate_file
    os.system(cmd)

    return out_len_annotate_file


def merge_bed(input_bed_file: str, output_bed_file: str) -> None:
    """
    Merge overlapping regions in a BED file.

    Parameters
    ----------
    input_bed_file : str
        Path to the input BED file.
    output_bed_file : str
        Path to the output BED file.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.

    """

    if not os.path.exists(input_bed_file):
        raise FileNotFoundError(f"File '{input_bed_file}' does not exist.")

    pybedtools.BedTool(input_bed_file).merge().saveas(output_bed_file)


def intersect_bed_regions(
    include_regions: list[str],
    exclude_regions: list[str] = None,
    output_bed: str = "output.bed",
    assume_input_sorted: bool = False,
    max_mem: int = None,
    sp: SimplePipeline = None,
):
    """
    Intersect BED regions with the option to subtract exclude regions,
    using bedops for the operations (must be installed).

    Parameters
    ----------
    include_regions : list of str
        List of paths to BED files to be intersected.
    exclude_regions : list of str, optional
        List of paths to BED or VCF files to be subtracted from the intersected result.
    output_bed : str, optional
        Path to the output BED file.
    assume_input_sorted : bool, optional
        If True, assume that the input files are already sorted. If False, the function will sort them on-the-fly.
    max_mem : int, optional
        Maximum memory in bytes allocated for the sort-bed operations.
        If not specified, the function will allocate 80% of the available system memory.
    sp : SimplePipeline, optional
        SimplePipeline object to be used for printing and executing commands.

    Returns
    -------
    None
        The function saves the intersected (and optionally subtracted) regions to the output_bed file.

    Raises
    ------
    FileNotFoundError
        If any of the input files do not exist.

    """

    # Checking if all input files exist
    for region_file in include_regions + (exclude_regions if exclude_regions else []):
        if not os.path.exists(region_file):
            raise FileNotFoundError(f"File '{region_file}' does not exist.")

    # Make sure bedops is installed
    assert (
        subprocess.call(["bedops", "--version"]) == 0
    ), "bedops is not installed. Please install bedops and make sure it is in your PATH."

    # If only one include region is provided and no exclude regions, just copy the file to the output
    if len(include_regions) == 1 and exclude_regions is None and assume_input_sorted:
        shutil.copy(include_regions[0], output_bed)
        return

    # If max_mem is not specified, set it to 80% of available memory
    total_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    if max_mem:
        if max_mem < total_memory:
            logger.warning(
                f"max_mem ({max_mem}) cannot be larger than the total system memory ({total_memory}). "
                f"Using {int(total_memory * 0.8)}."
            )
            max_mem = int(total_memory * 0.8)
    else:
        max_mem = int(total_memory * 0.8)

    sort_bed_cmd = f"sort-bed --max-mem {max_mem}"
    with tempfile.TemporaryDirectory() as tempdir:
        # Function to get a temp file path within the tempdir
        def get_temp_file():
            return os.path.join(tempdir, next(tempfile._get_candidate_names()))

        # Process the include regions
        if len(include_regions) == 1:
            if not assume_input_sorted:
                sorted_include = get_temp_file()
                print_and_execute(
                    f"{sort_bed_cmd} {include_regions[0]} > {sorted_include}",
                    simple_pipeline=sp,
                    module_name=__name__,
                )
                intersected_include_file = sorted_include
            else:
                intersected_include_file = include_regions[0]
        else:
            sorted_includes = []
            for include_bed_or_vcf in include_regions:
                if not assume_input_sorted:
                    sorted_include = get_temp_file()
                    print_and_execute(
                        f"{sort_bed_cmd} {include_bed_or_vcf} > {sorted_include}",
                        simple_pipeline=sp,
                        module_name=__name__,
                    )
                    sorted_includes.append(sorted_include)
                else:
                    sorted_includes.append(include_bed_or_vcf)
            intersected_include = get_temp_file()
            print_and_execute(
                f"bedops --header --intersect {' '.join(sorted_includes)} > {intersected_include}",
                simple_pipeline=sp,
                module_name=__name__,
            )
            intersected_include_file = intersected_include

        # Process the exclude_regions similarly and get the subtracted regions
        if exclude_regions:
            excludes = []
            for exclude_bed_or_vcf in exclude_regions:
                sorted_exclude_bed = get_temp_file()
                if exclude_bed_or_vcf.endswith(".vcf") or exclude_bed_or_vcf.endswith(".vcf.gz"):  # vcf file
                    if not assume_input_sorted:
                        print_and_execute(
                            f"bcftools view {exclude_bed_or_vcf} | "
                            "bcftools annotate -x INFO,FORMAT | "
                            f"vcf2bed --max-mem {max_mem} > {sorted_exclude_bed}",
                            simple_pipeline=sp,
                            module_name=__name__,
                        )
                    else:
                        print_and_execute(
                            f"bcftools view {exclude_bed_or_vcf} | "
                            "bcftools annotate -x INFO,FORMAT | "
                            f"vcf2bed --do-not-sort > {sorted_exclude_bed}",
                            simple_pipeline=sp,
                            module_name=__name__,
                        )
                    excludes.append(sorted_exclude_bed)
                elif not assume_input_sorted:
                    print_and_execute(
                        f"{sort_bed_cmd} {exclude_bed_or_vcf} > {sorted_exclude_bed}",
                        simple_pipeline=sp,
                        module_name=__name__,
                    )
                    excludes.append(sorted_exclude_bed)
                else:  # bed file and assume_input_sorted
                    excludes.append(exclude_bed_or_vcf)

            # Construct the final command
            cmd = f"bedops --header --difference {intersected_include_file} {' '.join(excludes)} > {output_bed}"
        else:
            cmd = f"mv {intersected_include_file} {output_bed}"

        # Execute the final command
        print_and_execute(cmd, simple_pipeline=sp, module_name=__name__)


def count_bases_in_bed_file(file_path: str) -> int:
    """
    Count the number of bases in a given region from a file.

    Parameters
    ----------
    file_path : str
        Path to the bed file containing region data. interval_list files are also supported.

    Returns
    -------
    int
        Total number of bases in the provided region.

    Raises
    ------
    FileNotFoundError
        If the provided file path does not exist.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # count the # of bases in region
    n_bases_in_region = 0
    with open(file_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("@") and not line.startswith("#"):  # handle handles and interval_list files
                spl = line.rstrip().split("\t")
                n_bases_in_region += int(spl[2]) - int(spl[1])

    return n_bases_in_region
