import argparse
import logging
import os
import subprocess
from os.path import join as pjoin

ap = argparse.ArgumentParser(
    prog="create_somatic_gt_file.py",
    description="Create a ground truth file which is composed of tumor and normal files",
)
ap.add_argument(
    "--gt_tumor",
    help="Ground truth file of the tumor genome (vcf)",
    required=True,
    type=str,
)
ap.add_argument(
    "--gt_normal",
    help="Ground truth file of the normal genome (vcf)",
    required=True,
    type=str,
)
ap.add_argument(
    "--gt_tumor_name",
    help="Name of the tumor sample, just for output file names",
    required=True,
    type=str,
)
ap.add_argument(
    "--gt_normal_name",
    help="Name of the normal sample, just for output file names",
    required=True,
    type=str,
)
ap.add_argument(
    "--regions_bed",
    help="Regions to apply the simulation on (bed)",
    required=False,
    type=str,
)
ap.add_argument(
    "--cmp_intervals",
    help="Comparison regions bed file from which we will reduce " "the problematic positions (bed)",
    required=True,
    type=str,
)
ap.add_argument("--output_folder", help="Output folder", required=True, type=str)
args = ap.parse_args()

# EXPLANATION FOR THE SCRIPT:
# When running mutect of tumor normal and we want to make a comparison analysis, we need to
# compare the output from mutect to a gt_file which is the subtraction of the normal gt
# from the tumor gt for having the mutations from the tumor only.
# The subtraction is not so simple in positions which are variants of both of them.
# In these cases we want to remove only places which are with any similar alleles.
# In addition, places with deletions in both of them should also be deleted.
#
# Hence, in this script we delete these problematic places and create a bed file.
# This bed file should be used as the high_conf parameter in the run_comaprison_pipeline script.
# We also create the gt_vcf which is the subtraction of both of them

# output files to be used:
# OUTPUT_gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}.vcf.gz -  the gt vcf for comparison
# OUTPUT_{prefix_cmp_interval}_no_problematic_positions_in_regions_only.bed - the cmp_interval file for comparison
###


logger = logging.getLogger(__name__ if __name__ != "__main__" else "create_somatic_gt_file")
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# GT_VCF FILE
cmd = [
    "bcftools",
    "isec",
    "-p",
    args.output_folder,
    args.gt_tumor,
    args.gt_normal,
    "-Oz"
]
logger.info(" ".join(cmd))
subprocess.check_call(cmd)

cmd = [
    "mv",
    pjoin(args.output_folder, "0000.vcf.gz"),
    pjoin(
        args.output_folder,
        f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}.vcf.gz",
    ),
]
logger.info(" ".join(cmd))
subprocess.check_call(cmd)

cmd = [
    "mv",
    pjoin(args.output_folder, "0000.vcf.gz.tbi"),
    pjoin(
        args.output_folder,
        f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}.vcf.gz.tbi",
    ),
]
logger.info(" ".join(cmd))
subprocess.check_call(cmd)


# CMP_INTERVALS CREATION
# exact_match intersect
cmd = [
    "bcftools",
    "isec",
    "-p",
    args.output_folder,
    args.gt_tumor,
    args.gt_normal,
    "-Oz",
    "-c",
    "none",
]
logger.info(" ".join(cmd))
subprocess.check_call(cmd)
# 0002.vcf.gz for records from tumor shared by both
cmd = [
    "mv",
    pjoin(args.output_folder, "0002.vcf.gz"),
    pjoin(
        args.output_folder,
        f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_exact.vcf.gz",
    ),
]
logger.info(" ".join(cmd))
subprocess.check_call(cmd)

cmd = [
    "mv",
    pjoin(args.output_folder, "0002.vcf.gz.tbi"),
    pjoin(
        args.output_folder,
        f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_exact.vcf.gz.tbi",
    ),
]
logger.info(" ".join(cmd))
subprocess.check_call(cmd)


# position match intersect
cmd = [
    "bcftools",
    "isec",
    "-p",
    args.output_folder,
    args.gt_tumor,
    args.gt_normal,
    "-Oz",
    "-c",
    "all",
]
logger.info(" ".join(cmd))
subprocess.check_call(cmd)
# 0002.vcf.gz	for records from tumor shared by both
cmd = [
    "mv",
    pjoin(args.output_folder, "0002.vcf.gz"),
    pjoin(
        args.output_folder,
        f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_position.vcf.gz",
    ),
]
logger.info(" ".join(cmd))
subprocess.check_call(cmd)

cmd = [
    "mv",
    pjoin(args.output_folder, "0002.vcf.gz.tbi"),
    pjoin(
        args.output_folder,
        f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_position.vcf.gz.tbi",
    ),
]
logger.info(" ".join(cmd))
subprocess.check_call(cmd)


# position minus exact intersect
cmd = [
    "bcftools",
    "isec",
    "-p",
    args.output_folder,
    pjoin(
        args.output_folder,
        f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_position.vcf.gz",
    ),
    pjoin(
        args.output_folder,
        f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_exact.vcf.gz",
    ),
    "-Oz",
    "-c",
    "all",
]
logger.info(" ".join(cmd))
subprocess.check_call(cmd)
# 0000.vcf is for recodes from position only - no exact match
cmd = [
    "mv",
    pjoin(args.output_folder, "0000.vcf.gz"),
    pjoin(
        args.output_folder,
        f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_intersect_position_no_exact_match.vcf.gz",
    ),
]
logger.info(" ".join(cmd))
subprocess.check_call(cmd)

cmd = [
    "mv",
    pjoin(args.output_folder, "0000.vcf.gz.tbi"),
    pjoin(
        args.output_folder,
        f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_intersect_position_no_exact_match.vcf.gz.tbi",
    ),
]
logger.info(" ".join(cmd))
subprocess.check_call(cmd)


# create a bed file out of this vcf
cmd = [
    "gunzip",
    "-f",
    pjoin(
        args.output_folder,
        f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_intersect_position_no_exact_match.vcf.gz",
    ),
]
logger.info(" ".join(cmd))
subprocess.check_call(cmd)

cmd = ["convert2bed", "--input=vcf", "--output=bed"]
with open(pjoin(args.output_folder, f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_intersect_position_no_exact_match.vcf"), "rb",) as fd, open(
    pjoin(args.output_folder, f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_intersect_position_no_exact_match.bed"),
    "w",
    encoding="utf-8",
) as outfile:
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd, stdin=fd, stdout=outfile)

# deletions in a separate bed file for the deletion because they are not converted with the correct positions
cmd = ["convert2bed", "--input=vcf", "--output=bed", "--deletions"]
with open(pjoin(args.output_folder, f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_intersect_position_no_exact_match.vcf"), "rb",) as fd, open(
    pjoin(
        args.output_folder,
        f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_intersect_position_no_exact_match_deletions.bed",
    ),
    "w",
    encoding="utf-8",
) as outfile:
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd, stdin=fd, stdout=outfile)


# union the 2 bed files
cmd = [
    "cat",
    pjoin(args.output_folder, f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_intersect_position_no_exact_match.bed"),
    pjoin(
        args.output_folder,
        f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_intersect_position_no_exact_match_deletions.bed",
    ),
]
with open(
    pjoin(
        args.output_folder,
        f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_intersect_position_no_exact_match_with_deletion.bed",
    ),
    "w",
    encoding="utf-8",
) as outfile:
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd, stdout=outfile)

# sort the bed file
cmd = [
    "bedtools",
    "sort",
    "-i",
    pjoin(
        args.output_folder,
        f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_intersect_position_no_exact_match_with_deletion.bed",
    ),
]
with open(
    pjoin(
        args.output_folder,
        f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_intersect_position_no_exact_match_with_deletion_sorted.bed",
    ),
    "w",
    encoding="utf-8",
) as outfile:
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd, stdout=outfile)


# new high conf bed file without these problematic positions
prefix_cmp_interval = os.path.splitext(os.path.basename(args.cmp_intervals))[0].split(".")[0]
cmd = [
    "bedtools",
    "subtract",
    "-a",
    args.cmp_intervals,
    "-b",
    pjoin(
        args.output_folder,
        f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_intersect_position_no_exact_match_with_deletion_sorted.bed",
    ),
]
outputfile = (
    f"OUTPUT_{prefix_cmp_interval}_no_problematic_positions.bed"
    if args.regions_bed is None
    else f"{prefix_cmp_interval}_no_problematic_positions.bed"
)
with open(pjoin(args.output_folder, outputfile), "w", encoding="utf-8") as outfile:
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd, stdout=outfile)


if args.regions_bed is not None:
    cmd = [
        "bedtools",
        "intersect",
        "-a",
        pjoin(args.output_folder, f"{prefix_cmp_interval}_no_problematic_positions.bed"),
        "-b",
        args.regions_bed,
    ]
    with open(
        pjoin(
            args.output_folder,
            f"OUTPUT_{prefix_cmp_interval}_no_problematic_positions_in_regions_only.bed",
        ),
        "w",
        encoding="utf-8",
    ) as outfile:
        logger.info(" ".join(cmd))
        subprocess.check_call(cmd, stdout=outfile)



## het only variants
file_path =     pjoin(
        args.output_folder,
        f"gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}.vcf.gz",
    )
output_path =     pjoin(
        args.output_folder,
        f'OUTPUT_gt_{args.gt_tumor_name}_minus_{args.gt_normal_name}_het_only.vcf.gz',
    )
import pysam
new_sample_name = f"{args.gt_tumor_name}_minus_{args.gt_normal_name}"
vcf_in = pysam.VariantFile(file_path,"r")
header = vcf_in.header.copy()
header.add_sample(new_sample_name)
vcf_out = pysam.VariantFile(output_path, "w", header=header)

dd = []
for read in vcf_in:
    dd.append(read.samples[0]['GT'])
    if read.samples[0]['GT'] == (1,1):
        read.samples[0]['GT'] = (0,1)
    vcf_out.write(read)
vcf_in.close()
vcf_out.close()

cmd = [
    "bcftools",
    "index",
    "-t",
    output_path,
]
logger.info(" ".join(cmd))
subprocess.check_call(cmd)