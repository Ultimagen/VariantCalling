import argparse
import os
from os.path import dirname

ap = argparse.ArgumentParser(prog="convert_graph_alignment_to_ug_format.py", description="merge bam file originated from vg graph alignment with original ug tags")
ap.add_argument(
    "--vg_graph_bam_file",
    help="path to vg_graph bam file.",
    type=str,
    required=True
)
ap.add_argument(
    "--ug_unmapped_bam_file",
    help="path to ug unmapped bam file sorted by query_name. to generate this file please use convert_bam_to_ubam.wdl",
    type=str,
    required=True
)
ap.add_argument(
    "--reference_file",
    help="path to reference fasta file",
    type=str,
    required=True
)
ap.add_argument(
    "--read_group_string",
    help="read group string to add to vg-unammaped-bwa-aligned",
    type=str,
    required=True
)
ap.add_argument(
    "--read_group_id",
    help="read group id to update in final ",
    type=str,
    required=True
)
ap.add_argument('--align_vg_unmapped', action='store_true')
ap.add_argument('--dryrun', action='store_true')


args = ap.parse_args()

print('')
print("## ReorderSam for vg_graph_aligned bam in order to replace header (to fit broad's reference)")
command = 'picard ReorderSam VALIDATION_STRINGENCY=SILENT ' + \
          'INPUT=' + args.vg_graph_bam_file + \
          ' OUTPUT=' + args.vg_graph_bam_file + '.Reorder.bam' + \
          ' REFERENCE=' + args.reference_file + \
          ' > stdout.picard_ReorderSam 2> stderr.picard_ReorderSam'
print(command)
if not args.dryrun:
    os.system(command)

print('## fetch unmapped reads from original file')
command = 'nohup samtools view -b -f 4 -o ' + args.vg_graph_bam_file + '.Unmapped.bam ' + args.vg_graph_bam_file
print(command)
if not args.dryrun:
    os.system(command)

print('## concat vg_graph_aligned reordered and unmapped')
command = 'samtools merge -@ 14 ' + args.vg_graph_bam_file + '.Reorder.withUnmapped.bam ' + args.vg_graph_bam_file + '.Reorder.bam ' + args.vg_graph_bam_file + '.Unmapped.bam' + ' > stdout.samtools_merge 2> stderr.samtools_merge'
print(command)
if not args.dryrun:
    os.system(command)

print('## qsort merged-reordered bam')
command = 'samtools sort -@ 14 -n -O bam -o ' + args.vg_graph_bam_file + '.Reorder.withUnmapped.qsorted.bam ' + args.vg_graph_bam_file + '.Reorder.withUnmapped.bam' + ' > stdout.samtools_qsort_afterReorder 2> stderr.samtools_qsort_afterReorder '
print(command)
if not args.dryrun:
    os.system(command)

print ('# MergeBamAlignment graph-aligned with ug-unmapped to attach ug-tags')
command = 'picard MergeBamAlignment VALIDATION_STRINGENCY=SILENT EXPECTED_ORIENTATIONS=FR ATTRIBUTES_TO_RETAIN=X0 ATTRIBUTES_TO_RETAIN=tm ' \
          'ATTRIBUTES_TO_RETAIN=tf ATTRIBUTES_TO_RETAIN=RX ATTRIBUTES_TO_REMOVE=NM ATTRIBUTES_TO_REMOVE=MD ATTRIBUTES_TO_REVERSE=ti ' \
          'ATTRIBUTES_TO_REVERSE=tp ATTRIBUTES_TO_REVERSE=t0 ' \
          'ALIGNED_BAM=' + args.vg_graph_bam_file + '.Reorder.withUnmapped.qsorted.bam' \
          ' UNMAPPED_BAM=' + args.ug_unmapped_bam_file + \
          ' OUTPUT=' + args.vg_graph_bam_file + '.with_ug_info.bam' + \
          ' REFERENCE_SEQUENCE=' + args.reference_file + \
          ' SORT_ORDER="queryname" IS_BISULFITE_SEQUENCE=false CLIP_ADAPTERS=false ' \
          'ALIGNED_READS_ONLY=false MAX_RECORDS_IN_RAM=2000000 ADD_MATE_CIGAR=true ' \
          'MAX_INSERTIONS_OR_DELETIONS=-1 PRIMARY_ALIGNMENT_STRATEGY=MostDistant ' \
          'UNMAPPED_READ_STRATEGY=COPY_TO_TAG ALIGNER_PROPER_PAIR_FLAGS=true ' \
          'UNMAP_CONTAMINANT_READS=false ADD_PG_TAG_TO_READS=false ' \
          '> stdout.picard_MergeBamAlignment 2> stderr.picard_MergeBamAlignment'
print(command)
if not args.dryrun:
    os.system(command)

print('# add read group')
#'UGAv3-20-E351E76'
command = 'samtools addreplacerg -R ' + args.read_group_id +\
          ' -o ' + args.vg_graph_bam_file + '.with_ug_info.withrg.bam '+\
          ' -@14 --verbosity 3 -O BAM ' + args.vg_graph_bam_file + '.with_ug_info.bam'
print(command)
if not args.dryrun:
    os.system(command)


if args.align_vg_unmapped:
    print('')
    print('##### Align graph-unmapped back to hg38 #####')
    print('# convert bam to fastq')
    command = 'picard SamToFastq INPUT=' + args.vg_graph_bam_file + '.Unmapped.bam' + ' FASTQ=' + args.vg_graph_bam_file + '.Unmapped.fq ' + 'INTERLEAVE=true NON_PF=true > stdout.picard_SamToFastq 2> stderr.picard_SamToFastq '
    print(command)
    if not args.dryrun:
        os.system(command)

    print('# align unmapped reads')
    command = 'bwa mem -K 100000000 -p -v 3 -t 12 -Y /data/ref_genomes/gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta ' + args.vg_graph_bam_file + '.Unmapped.fq' + ' > ' + args.vg_graph_bam_file + '.vgUnmapped.bwa_aligned.sam 2> stderr.bwa_mem_vg-unmapped'
    print(command)
    if not args.dryrun:
        os.system(command)

    print('# convert sam to bam')
    command = 'samtools view -@ 14 -b -o ' + args.vg_graph_bam_file + '.vgUnmapped.bwa_aligned.bam ' + args.vg_graph_bam_file + '.vgUnmapped.bwa_aligned.sam'
    print(command)
    if not args.dryrun:
        os.system(command)

    print('# add read group to aligned vg_unmapped')
    # rgstrg='@RG\tID:UGAv3-20-E351E76\tDT:2022-05-19T17:47:23+0000\tPL:LS454\tSM:UGAv3-20\tPU:004777_1.20220519.CATCCTGCATCGCAGAT\tmc:12\tBC:CATCCTGCATCGCAGAT\tFO:TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA\tPM:V1.3\tvn:f577cf19 2022-05-09 12:38:25+0000\ttf:1.3\ttq:1\tLB:UGAv3-20\ttp:reference'

    command = 'samtools addreplacerg -r ' + args.read_group_string + \
              ' -o ' + args.vg_graph_bam_file + '.vgUnmapped.bwa_aligned.withrg.bam ' + \
              '-@10 --verbosity 3 -O BAM ' + args.vg_graph_bam_file + '.vgUnmapped.bwa_aligned.bam'
    print(command)
    if not args.dryrun:
        os.system(command)

    print('# sort aligned vg_unmapped')
    command = 'samtools sort -@14 -n -o ' + args.vg_graph_bam_file + '.vgUnmapped.bwa_aligned.withrg.qsort.bam ' + args.vg_graph_bam_file + '.vgUnmapped.bwa_aligned.withrg.bam'
    print(command)
    if not args.dryrun:
        os.system(command)