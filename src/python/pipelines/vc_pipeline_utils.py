import subprocess
import configargparse
import time
from os.path import join as pjoin
from os.path import basename, dirname, splitext
import python.utils as utils
import os
import re
import pandas as pd
import numpy as np
import pysam


def parse_params_file(pipeline_name):
    ap = configargparse.ArgParser()
    ap.add('-c', required=True, is_config_file=True, help='config file path')
    group1 = ap.add_mutually_exclusive_group(required=True)
    group1.add('--DataFileName', help='Path + prefix of the output_files')
    group1.add('--em_vc_output_dir', help='Output dir')
    ap.add('--em_vc_genome', required=True,
           help='Path to genome file (bwa index, dict should exist)')
    ap.add('--em_vc_number_of_cpus', required=False, help='Number of CPUs on the machine',
           type=int,
           default=12)
    ap.add('--em_vc_rerun_all', default=False, action='store_true')
    if pipeline_name == 'error_metrics':
        ap.add('--em_vc_demux_file', help='Path to the demultiplexed bam')
        ap.add('--em_vc_number_to_sample', required=False, help='Number of records to downsample',
               type=int)
        ap.add('--em_vc_chromosomes_list', required=False,
               help='File with the list of chromosomes to test')
    elif pipeline_name == 'rapidqc':
        ap.add('--rqc_demux_file', help='Path to the demultiplexed bam')
        ap.add('--rqc_chromosome', help='Single chromosome to filter for', required=True,
               type=str)
        ap.add('--rqc_evaluation_intervals_names', required=False,
               help='Comma separated list of evaluation interval names',
               type=str)
        ap.add('--rqc_evaluation_intervals', required=False,
               help='Comma separated list of evaluation interval file names',
               type=str)
        ap.add('--rqc_coverage_intervals_table', required=False,
               help='File with TSV coverage evaluation intervals',
               type=str)
        ap.add('--rqc_coverage_intervals_location', required=True,
               help='Path to coverage interval locations (prepended to interval_names in intervals_table',
               type=str, default='./')
        ap.add('--rqc_cram_reference_file', required=False, 
            help='Reference fasta used for CRAM compression (if demux_file is CRAM)', default=None)
        ap.add('--rqc_disable_alignment', required=False, 
            help='Do not realign the CRAM', default=False, action='store_true')

    elif pipeline_name == 'variant_calling':
        ap.add('--em_vc_demux_file', help='Path to the demultiplexed bam')
        ap.add('--em_vc_recalibration_model', required=False,
               help='recalibration model (h5)')
        ap.add('--em_vc_ground_truth', required=False, help='Ground truth file to compare',
               type=str)
        ap.add('--em_vc_ground_truth_highconf', required=False, help='Ground truth high confidence file',
               type=str)
        ap.add('--em_vc_gaps_hmers_filter', required=False, help='Bed file with regions to filter out',
               type=str)
        ap.add('--em_vc_chromosomes_list', required=False,
               help='File with the list of chromosomes to test')
    else:
        raise RuntimeError(f"{pipeline_name} is not a defined pipeline")
    args = ap.parse_known_args()[0]
    if pipeline_name == 'rapidqc':
        args.em_vc_demux_file = args.rqc_demux_file
        args.em_vc_basename = basename(args.rqc_demux_file)
        args.rqc_evaluation_intervals_names = [
            x.strip() for x in args.rqc_evaluation_intervals_names.split(',')]
        args.rqc_evaluation_intervals = [
            x.strip() for x in args.rqc_evaluation_intervals.split(',')]
        assert len(args.rqc_evaluation_intervals) == len(
            args.rqc_evaluation_intervals_names), 'Different length of names and evaluation intervals given'
    if args.DataFileName is not None:
        args.em_vc_output_dir = dirname(args.DataFileName)
        args.em_vc_basename = basename(args.DataFileName)
    else:
        args.em_vc_basename = basename(args.em_vc_demux_file)
    return args

def extract_total_n_reads(input_file, output_file, nthreads) : 
    output_count, output_err = output_file
    cmd = ['samtools', 'view', '-c', f'-@{nthreads}', input_file[0]]
    with open(output_count, 'w') as out, open(output_err, 'w') as err:
        err.write(" ".join(cmd))
        err.write("\n")
        err.flush()
        subprocess.check_call(cmd, stderr=err, stdout=out)


def head_file(input_file, output_file, number_to_sample, nthreads):
    output_bam, output_err = output_file
    cmd1 = ['samtools', 'view',
            '-@%d' % max(1, int((nthreads - 2) / 2)), '-h', input_file]
    cmd2 = [
        'head', f"-{number_to_sample + 100}"]
    cmd3 = [
        'samtools', 'view',
        '-@%d' % max(1, int((nthreads - 2) / 2)), '-b', '-o', output_bam, '-']
    with open(output_err, 'w') as output_err_handle:
        output_err_handle.write(' '.join(cmd1))
        output_err_handle.flush()
        output_err_handle.write(' | ')
        output_err_handle.write(' '.join(cmd2))
        output_err_handle.flush()
        output_err_handle.write(' | ')
        output_err_handle.write(' '.join(cmd3))
        output_err_handle.write('\n')
        output_err_handle.flush()
        task1 = subprocess.Popen(cmd1,
                                 stdout=(subprocess.PIPE), stderr=output_err_handle)
        task2 = subprocess.Popen(cmd2,
                                 stdin=(task1.stdout), stdout=(subprocess.PIPE), stderr=output_err_handle)
        task1.stdout.close()
        task3 = subprocess.Popen(cmd3,
                                 stdin=(task2.stdout), stderr=output_err_handle)
        task2.stdout.close()
        _ = task3.communicate()
    time.sleep(30)
    exception_string = ''
    flag = False
    time.sleep(30)
    task1.poll()
    task2.poll()
    if task1.returncode != 0:
        if task1.returncode != -13:
            exception_string += f"bam->sam failed: rc={task1.returncode} "
            flag = True
    if task2.returncode != 0:
        exception_string += f"head -{number_to_sample} failed: rc={task2.returncode} "
        flag = True
    if task3.returncode != 0:
        flag = True
        exception_string += f"sam->bam failed: rc={task3.returncode}"
    if flag:
        raise RuntimeError(exception_string)


def align(input_file, output_file, genome_file, nthreads):
    output_bam, output_err = output_file
    output_err_handle = open(output_err, 'w')
    nthreads_alignment = max(1, (nthreads - 2))
    if type(input_file) == list or type(input_file) == tuple:
        cmd1 = [
            'picard', '-Xms5000m', 'SamToFastq',
            'INPUT=%s' % input_file[0], 'FASTQ=/dev/stdout']
    else:
        cmd1 = [
            'picard', '-Xms5000m', 'SamToFastq',
            'INPUT=%s' % input_file, 'FASTQ=/dev/stdout']
    output_err_handle.write(' '.join(cmd1))
    output_err_handle.flush()
    task1 = subprocess.Popen(cmd1,
                             stdout=(subprocess.PIPE), stderr=output_err_handle)
    cmd2 = ['bwa', 'mem', '-t', str(nthreads_alignment), genome_file, '-']
    output_err_handle.write(' | ')
    output_err_handle.write(' '.join(cmd2))
    output_err_handle.flush()
    task2 = subprocess.Popen(cmd2, stdin=(task1.stdout), stdout=(subprocess.PIPE),
                             stderr=output_err_handle)
    task1.stdout.close()
    if type(input_file) == list or type(input_file) == tuple:
        cmd3 = [
            'picard', '-Xms3000m', 'MergeBamAlignment', 'VALIDATION_STRINGENCY=SILENT',
            'ATTRIBUTES_TO_RETAIN=X0', 'ATTRIBUTES_TO_REMOVE=NM', 'ATTRIBUTES_TO_REMOVE=MD',
            'ALIGNED_BAM=/dev/stdin', 'UNMAPPED_BAM=%s' % input_file[
                0], 'OUTPUT=%s' % output_bam,
            'REFERENCE_SEQUENCE=%s' % genome_file, 'PAIRED_RUN=false', 'SORT_ORDER="unsorted"',
            'IS_BISULFITE_SEQUENCE=false', 'ALIGNED_READS_ONLY=false', 'CLIP_ADAPTERS=false',
            'MAX_RECORDS_IN_RAM=2000000', 'MAX_INSERTIONS_OR_DELETIONS=-1', 'PRIMARY_ALIGNMENT_STRATEGY=MostDistant',
            'UNMAP_CONTAMINANT_READS=true', 'ADD_PG_TAG_TO_READS=false']
    else:
        cmd3 = [
            'picard', '-Xms3000m', 'MergeBamAlignment', 'VALIDATION_STRINGENCY=SILENT',
            'ATTRIBUTES_TO_RETAIN=X0', 'ATTRIBUTES_TO_REMOVE=NM', 'ATTRIBUTES_TO_REMOVE=MD',
            'ALIGNED_BAM=/dev/stdin', 'UNMAPPED_BAM=%s' % input_file, 'OUTPUT=%s' % output_bam,
            'REFERENCE_SEQUENCE=%s' % genome_file, 'PAIRED_RUN=false', 'SORT_ORDER="unsorted"',
            'IS_BISULFITE_SEQUENCE=false', 'ALIGNED_READS_ONLY=false', 'CLIP_ADAPTERS=false',
            'MAX_RECORDS_IN_RAM=2000000', 'MAX_INSERTIONS_OR_DELETIONS=-1', 'PRIMARY_ALIGNMENT_STRATEGY=MostDistant',
            'UNMAP_CONTAMINANT_READS=true', 'ADD_PG_TAG_TO_READS=false']

    output_err_handle.write(' | ')
    output_err_handle.write(' '.join(cmd3))
    output_err_handle.write('\n')
    output_err_handle.flush()
    task3 = subprocess.Popen(cmd3, stdin=(task2.stdout), stdout=output_err_handle,
                             stderr=output_err_handle)
    task2.stdout.close()
    _ = task3.communicate()
    output_err_handle.close()
    exception_string = ''
    flag = False
    time.sleep(30)
    task1.poll()
    task2.poll()
    if task1.returncode != 0:
        exception_string += f"SamToFastq failed: rc={task1.returncode} "
        flag = True
    if task2.returncode != 0:
        exception_string += f"Alignment failed: rc={task2.returncode} "
        flag = True
    if task3.returncode != 0:
        flag = True
        exception_string += f"sam->bam failed: rc={task3.returncode}"
    if flag:
        raise RuntimeError(exception_string)


def align_and_merge(input_file, output_file, genome_file, nthreads):
    output_bam, output_err = output_file
    output_err_handle = open(output_err, 'w')
    nthreads_alignment = max(1, int(nthreads - 3))
    cmd1 = [
        'picard', '-Xms5000m', 'SamToFastq',
        'INPUT=%s' % input_file, 'FASTQ=/dev/stdout']
    output_err_handle.write(' '.join(cmd1))
    output_err_handle.flush()
    task1 = subprocess.Popen(cmd1,
                             stdout=(subprocess.PIPE), stderr=output_err_handle)
    cmd2 = ['bwa', 'mem', '-t', str(nthreads_alignment), genome_file, '-']
    output_err_handle.write(' | ')
    output_err_handle.write(' '.join(cmd2))
    output_err_handle.flush()
    task2 = subprocess.Popen(cmd2, stdin=(task1.stdout), stdout=(subprocess.PIPE),
                             stderr=output_err_handle)
    task1.stdout.close()
    cmd3 = [
        'picard', '-Xms3000m', 'MergeBamAlignment', 'VALIDATION_STRINGENCY=SILENT',
        'ATTRIBUTES_TO_RETAIN=X0', 'ATTRIBUTES_TO_REMOVE=NM', 'ATTRIBUTES_TO_REMOVE=MD',
        'ALIGNED_BAM=/dev/stdin', 'UNMAPPED_BAM=%s' % input_file, 'OUTPUT=%s' % output_bam,
        'REFERENCE_SEQUENCE=%s' % genome_file, 'PAIRED_RUN=false', 'SORT_ORDER="unsorted"',
        'IS_BISULFITE_SEQUENCE=false', 'ALIGNED_READS_ONLY=true', 'CLIP_ADAPTERS=false',
        'MAX_RECORDS_IN_RAM=2000000', 'MAX_INSERTIONS_OR_DELETIONS=-1', 'PRIMARY_ALIGNMENT_STRATEGY=MostDistant',
        'UNMAP_CONTAMINANT_READS=true', 'ADD_PG_TAG_TO_READS=false']
    output_err_handle.write(' | ')
    output_err_handle.write(' '.join(cmd3))
    output_err_handle.write('\n')
    output_err_handle.flush()
    task3 = subprocess.Popen(cmd3, stdin=(task2.stdout), stdout=output_err_handle,
                             stderr=output_err_handle)
    task2.stdout.close()
    _ = task3.communicate()
    output_err_handle.close()
    time.sleep(30)
    exception_string = ''
    flag = False
    task1.poll()
    task2.poll()
    if task1.returncode != 0:
        exception_string += f"SamToFastq failed: rc={task1.returncode} "
        flag = True
    if task2.returncode != 0:
        exception_string += f"Alignment failed: rc={task2.returncode} "
        flag = True
    if task3.returncode != 0:
        flag = True
        exception_string += f"MBA failed: rc={task3.returncode}"
    if flag:
        raise RuntimeError(exception_string)


def select_chromosome(input_file, output_files, nthreads, the_chromosome, cram_reference_fname=None):
    output_bam, output_err = output_files
    input_file = input_file

    if input_file.endswith('cram') and cram_reference_fname is not None:
        crammode = True
    elif input_file.endswith('cram') and cram_reference_fname is None:
        raise RuntimeError("Reference should be supplied for CRAM file")
    else:
        crammode = False

    with open(output_err, 'w') as outlog :
        samtools_in_threads = max(1, int(0.5 * nthreads))
        samtools_out_threads = max(1, int(0.5 * nthreads))        

        if not crammode:
            cmd1 = ['samtools', 'view', '-h', '-@',
                    str(samtools_in_threads), input_file]
        else:
            cmd1 = ['samtools', 'view', '-h', '-@',
                    str(samtools_in_threads), '--reference',
                    cram_reference_fname, input_file]

        cmd2 = ['awk', f'($3=="{the_chromosome}") || ($1 ~ /^@/)']

        cmd3 = [ 'samtools', 'view', f'-@{samtools_out_threads}', '-b','-o', output_bam, '-']

        cmd1_str = ' '.join(cmd1)
        cmd2_str = ' '.join(cmd2)
        cmd3_str = ' '.join(cmd3)
        outlog.write(f'{cmd1_str} | {cmd2_str} | {cmd3_str}' + "\n")
        outlog.flush()

        task1 = subprocess.Popen(cmd1, stdout=(subprocess.PIPE), stderr=outlog)
        task2 = subprocess.Popen(cmd2,
                                 stdout=(subprocess.PIPE), stderr=outlog, stdin=(task1.stdout))
        task1.stdout.close()
        task3 = subprocess.Popen(cmd3,stderr=outlog, stdin=(task2.stdout))
        task2.stdout.close()
        _ = task3.communicate()    
    # Collect results and RCs
    taskNames = ['extract', 'filter', 'compress']
    time.sleep(30)
    for x in [task1, task2, task3]:
        x.poll()

    rcs = [x.returncode for x in [task1, task2, task3]]
    for i in range(len(rcs)):
        result = ''
        if rcs[i] != 0:
            result += f"Task{i + 1}: {taskNames[i]} = {rcs[i]} "

    if len(result) > 0:
        raise RuntimeError(f"{result} of alignment failed")

def align_minimap_and_filter(input_file, output_files, genome_file, nthreads, the_chromosome, cram_reference_fname=None):
    output_bam, output_err = output_files
    input_file = input_file

    if input_file.endswith('cram') and cram_reference_fname is not None:
        alntmp = pysam.AlignmentFile(
            input_file, "rc", check_sq=False, reference_filename=cram_reference_fname)
        crammode = True
    elif input_file.endswith('cram') and cram_reference_fname is None:
        raise RuntimeError("Reference should be supplied for CRAM file")
    else:
        alntmp = pysam.AlignmentFile(input_file, check_sq=False)
        crammode = False

    header = str(alntmp.header).split('\n')
    rg_line = [x for x in header if x.startswith('@RG')]
    assert len(rg_line) == 1, 'uBAM does not contain RG or it is not single'
    rg_line = rg_line[0]
    rg_line = rg_line.replace('\t', '\\t')
    alntmp.close()

    with open(output_err, 'w') as outlog:
        if crammode:
            fifoname = f'{output_bam}.fifo.bam'
            if not os.path.exists(fifoname):
                os.mkfifo(fifoname)

        minimap_threads = max(1, int(0.8 * nthreads))
        samtools_in_threads = max(1, int(0.2 * nthreads))
        if not crammode:
            cmd1 = ['samtools', 'fastq', '-@',
                    str(samtools_in_threads), input_file]
        else:
            cmd1 = ['samtools', 'fastq', '-@',
                    str(samtools_in_threads), '--reference',
                    cram_reference_fname, input_file]

        cmd2 = ['minimap2', '-t', str(minimap_threads), '-R',
                rg_line, '-x', 'sr', '-y', '-a', genome_file, '-']
        cmd3 = ['awk', f'($3=="{the_chromosome}") || ($1 ~ /^@/)']

        if crammode:
            cmd4 = ['picard', '-Xms3000m', 'MergeBamAlignment', 'VALIDATION_STRINGENCY=SILENT',
                    'ATTRIBUTES_TO_RETAIN=X0', 'ATTRIBUTES_TO_REMOVE=NM', 'ATTRIBUTES_TO_REMOVE=MD',
                    'ALIGNED_BAM=/dev/stdin', f'UNMAPPED_BAM={fifoname}', 'OUTPUT=%s' % output_bam,
                    'REFERENCE_SEQUENCE=%s' % genome_file, 'PAIRED_RUN=false', 'SORT_ORDER="unsorted"',
                    'IS_BISULFITE_SEQUENCE=false', 'ALIGNED_READS_ONLY=true', 'CLIP_ADAPTERS=false',
                    'MAX_RECORDS_IN_RAM=2000000', 'MAX_INSERTIONS_OR_DELETIONS=-1',
                    'PRIMARY_ALIGNMENT_STRATEGY=MostDistant',
                    'UNMAP_CONTAMINANT_READS=true', 'ADD_PG_TAG_TO_READS=false']
        else:
            cmd4 = ['picard', '-Xms3000m', 'MergeBamAlignment', 'VALIDATION_STRINGENCY=SILENT',
                    'ATTRIBUTES_TO_RETAIN=X0', 'ATTRIBUTES_TO_REMOVE=NM', 'ATTRIBUTES_TO_REMOVE=MD',
                    'ALIGNED_BAM=/dev/stdin', 'UNMAPPED_BAM=%s' % input_file, 'OUTPUT=%s' % output_bam,
                    'REFERENCE_SEQUENCE=%s' % genome_file, 'PAIRED_RUN=false', 'SORT_ORDER="unsorted"',
                    'IS_BISULFITE_SEQUENCE=false', 'ALIGNED_READS_ONLY=true', 'CLIP_ADAPTERS=false',
                    'MAX_RECORDS_IN_RAM=2000000', 'MAX_INSERTIONS_OR_DELETIONS=-1',
                    'PRIMARY_ALIGNMENT_STRATEGY=MostDistant',
                    'UNMAP_CONTAMINANT_READS=true', 'ADD_PG_TAG_TO_READS=false']

        outlog.write(' | '.join((' '.join(cmd1), ' '.join(cmd2),
                                 ' '.join(cmd3), ' '.join(cmd4))) + '\n')
        outlog.flush()

        task1 = subprocess.Popen(cmd1, stdout=(subprocess.PIPE), stderr=outlog)
        task2 = subprocess.Popen(cmd2,
                                 stdout=(subprocess.PIPE), stderr=outlog, stdin=(task1.stdout))
        task1.stdout.close()
        task3 = subprocess.Popen(cmd3,
                                 stdout=(subprocess.PIPE), stderr=outlog, stdin=(task2.stdout))
        task2.stdout.close()
        task4 = subprocess.Popen(cmd4,
                                 stdout=outlog, stderr=outlog, stdin=(task3.stdout))
        task3.stdout.close()

        if crammode:
            cmd_cram1 = ['samtools', 'view', '-h', '-b',
                         '-T', cram_reference_fname, input_file]
            fifoout = open(fifoname, 'w')
            cmd_cram2 = ['picard', 'RevertSam', 'I=/dev/stdin',
                         f'O={fifoname}', 'MAX_DISCARD_FRACTION=0.005', 'ATTRIBUTE_TO_CLEAR=XT',
                         'ATTRIBUTE_TO_CLEAR=XN', 'ATTRIBUTE_TO_CLEAR=AS', 'ATTRIBUTE_TO_CLEAR=OC',
                         'ATTRIBUTE_TO_CLEAR=OP', 'REMOVE_DUPLICATE_INFORMATION=true', 'REMOVE_ALIGNMENT_INFORMATION=true',
                         'VALIDATION_STRINGENCY=LENIENT', 'SO=unsorted']
            outlog.write(' | '.join((' '.join(cmd_cram1), ' '.join(cmd_cram2))) + f">{fifoname}" + '\n')
            outlog.flush()
            task_cram1 = subprocess.Popen(
                cmd_cram1, stdout=(subprocess.PIPE), stderr=outlog)
            task_cram2 = subprocess.Popen(
                cmd_cram2, stdin=task_cram1.stdout, stdout=fifoout, stderr=outlog)
            task_cram1.stdout.close()

            task_cram2.wait()
        _ = task4.communicate()

    # Collect results and RCs
    taskNames = ['SamToFastq', 'minimap2', 'filter', 'addTags']
    time.sleep(30)
    for x in [task1, task2, task3]:
        x.poll()

    rcs = [x.returncode for x in [task1, task2, task3, task4]]

    if crammode:
        for x in [task_cram1, task_cram2]:
            x.poll()

        rcs += [x.returncode for x in [task_cram1, task_cram2]]
        taskNames += ['CramToBam', 'RevertSam']

    for i in range(len(rcs)):
        result = ''
        if rcs[i] != 0:
            result += f"Task{i + 1}: {taskNames[i]} = {rcs[i]} "

    if len(result) > 0:
        raise RuntimeError(f"{result} of alignment failed")
    fifoout.close()
    os.unlink(fifoout.name)


def prepare_fetch_intervals(input_file, output_file, genome_file):
    genome_dct = utils.get_chr_sizes(genome_file + '.sizes')
    with open(output_file, 'w') as output_handle:
        with open(input_file) as input_handle:
            for chrom in input_handle:
                if chrom:
                    output_handle.write(f"{chrom.strip()}\t0\t{genome_dct[chrom.strip()]}\n")


def filter_quality(input_file, output_files, nthreads):
    output_bam, output_err = output_files
    input_file = input_file[0]
    cmd = ['samtools', 'view', '-q20',
           '-@%d' % max(1, nthreads - 1), '-b', '-o', output_bam, input_file]
    with open(output_err, 'w') as output_err_handle:
        subprocess.check_call(cmd, stdout=output_err_handle,
                              stderr=output_err_handle)


def fetch_intervals(input_file, output_files):
    output_bam, output_err = output_files
    filter_intervals = input_file[1]
    input_file = input_file[0][0]
    cmd1 = ['bedtools', 'intersect', '-wa',
            '-a', input_file, '-b', filter_intervals]
    output_err_handle = open(output_err, 'w')
    output_file_handle = open(output_bam, 'w')
    subprocess.check_call(cmd1, stdout=output_file_handle,
                          stderr=output_err_handle)
    output_file_handle.close()
    output_err_handle.close()


def sort_file(input_file, output_file, nthreads, cram_reference_fname=None):
    output_bam, output_err = output_file
    if type(input_file)==str:
        input_file = [input_file]
    if cram_reference_fname is None :
        with open(output_err, 'w') as output_err_handle:

            cmd1 = [
                'samtools', 'sort',
                '-@%d' % max(1, nthreads - 1), '-T', dirname(output_bam), '-o', output_bam, input_file[0]]
            output_err_handle.write(" ".join(cmd1))
            output_err_handle.flush()
            subprocess.check_call(cmd1, stderr=output_err_handle)
    else: 
        with open(output_err, 'w') as output_err_handle:

            cmd1 = [
                'samtools', 'sort', '--reference', cram_reference_fname, '-O', "BAM", 
                '-@%d' % max(1, nthreads - 1), '-T', dirname(output_bam), '-o', output_bam, input_file[0]]
            output_err_handle.write(" ".join(cmd1))
            output_err_handle.flush()
            subprocess.check_call(cmd1, stderr=output_err_handle)

def recalibrate_file(input_file, output_files, recalibration_model, nthreads):
    input_file = input_file[0]
    output_bam, output_err = output_files
    pthreads = max(1, int(0.8 * nthreads))
    dthreads = max(1, int(0.1 * nthreads))
    cthreads = max(1, int(0.1 * nthreads))
    with open(output_err, 'w') as output_err_handle:
        cmd1 = [
            'LD_LIBRARY_PATH=/usr/local/lib',
            'recalibrate',
            '--input=%s' % input_file, '--binary',
            '--output=%s' % output_bam, '--model=%s' % recalibration_model,
            '--threshold=0.003', '--dthreads=%d' % dthreads, '--pthreads=%d' % pthreads, '--cthreads=%d' % cthreads]
        subprocess.check_call((' '.join(cmd1)),
                              stdout=output_err_handle, stderr=output_err_handle, shell=True)


def index_file(input_file, output_file):
    output_bam, output_err = output_file
    with open(output_err, 'w') as output_err_handle:
        cmd1 = [
            'samtools', 'index', input_file[0]]
        subprocess.check_call(cmd1, stderr=output_err_handle)


def create_variant_calling_intervals(input_file, output_file, n_threads):
    output_bed, output_err = output_file
    with open(output_bed, 'w') as output_bed_handle:
        with open(output_err, 'w') as output_err_handle:
            cmd1 = [
                'bedtools', 'makewindows', '-b',
                input_file, '-n', str(n_threads)]
            subprocess.check_call(cmd1,
                                  stdout=output_bed_handle, stderr=output_err_handle)


def split_intervals_into_files(input_file, output_files, output_file_name_root):
    for oo in output_files:
        os.unlink(oo)

    with open(input_file[0]) as input_file_handle:
        for i, line in enumerate(input_file_handle):
            with open(f"{output_file_name_root}.{i + 1}", 'w') as out:
                lsp = line.split()
                out.write('%s:%d-%d\n' %
                          (lsp[0], int(lsp[1]) + 1, int(lsp[2])))


def intersect_intervals(input_files: list, output_files: list):
    left, right = input_files
    output, log = output_files
    with open(log, 'w') as out:
        cmd = [
            'picard', 'IntervalListTools', f"I={left},{right}",
            'ACTION=INTERSECT', f"O={output}"]
        out.write(' '.join(cmd) + '\n')
        out.flush()
        subprocess.check_call(cmd, stdout=out, stderr=out)


def variant_calling(input_files, output_files, genome_file):
    aligned_bam, interval_file = input_files
    aligned_bam = aligned_bam[0]
    output_vcf, output_log = output_files
    gatk = 'gatk'
    my_env = os.environ.copy()
    interval = [x.strip() for x in open(interval_file)][0]
    cmd1 = [
        gatk, 'HaplotypeCaller', '-I', aligned_bam, '-O', output_vcf,
        '-R', genome_file, '--intervals', interval, '--likelihood-calculation-engine', 'FlowBased']
    with open(output_log, 'w') as output_log_handle:
        output_log_handle.write(' '.join(cmd1))
        output_log_handle.write('\n')
        task = subprocess.Popen(cmd1,
                                env=my_env, stdout=output_log_handle, stderr=output_log_handle)
        task.wait()
    if task.returncode != 0:
        raise RuntimeError(' '.join(cmd1) + ' exited abnormally')


def error_metrics(input_files, output_files, genome_file):
    aligned_bam, _ = input_files
    output_metrics, output_log = output_files
    cmd = ['picard', 'CollectAlignmentSummaryMetrics', 'R=%s' % genome_file,
           'I=%s' % aligned_bam, 'O=%s' % output_metrics, 'VALIDATION_STRINGENCY=LENIENT']
    with open(output_log, 'w') as output_err_handle:
        subprocess.check_call(cmd, stdout=output_err_handle,
                              stderr=output_err_handle)


def idxstats(input_files, output_files):
    aligned_bam, _ = input_files
    output_stats, output_log = output_files
    cmd = ['samtools', 'idxstats', aligned_bam]
    with open(output_log, 'w') as output_err_handle:
        with open(output_stats, 'w') as output_stats_handle:
            subprocess.check_call(cmd,
                                  stdout=output_stats_handle, stderr=output_err_handle)


def collect_alnstats(idxstats_file, filter_metrics):
    """
    Parameters
    ----------
    idxstats_file: str
        idxstats output
    filter_metrics: str
        Alignment metrics filter
    """
    df = pd.read_csv(idxstats_file, sep='\t', engine='python', header=None,
                     index_col=0,
                     names=['length', 'aligned_reads', 'unaligned_reads'])
    df = df.sum()
    df.drop(['length'], inplace=True)
    df1 = pd.read_csv(filter_metrics, sep='\t', comment='#',
                      engine='python').T[0]
    df['hq_aligned_reads'] = df1.loc['TOTAL_READS']
    df['total_reads'] = df['aligned_reads'] + df['unaligned_reads']
    df = df.loc[['total_reads', 'aligned_reads',
                 'hq_aligned_reads', 'unaligned_reads']]
    df.index = ['total', 'aligned', 'hq aligned', 'unaligned']
    df = pd.DataFrame(df)
    df.columns = ['Million reads']
    df['Million reads'] = (df['Million reads'] / 1000000.0).round(decimals=2)
    df['%'] = 100
    df.loc[('aligned', '%')] = np.round(df.loc[('aligned', 'Million reads')] / df.loc[('total',
                                                                                       'Million reads')] * 100, 2)
    df.loc[('unaligned', '%')] = np.round(df.loc[('unaligned', 'Million reads')] / df.loc[('total',
                                                                                           'Million reads')] * 100, 2)
    df.loc[('hq aligned', '%')] = np.round(df.loc[('hq aligned', 'Million reads')] / df.loc[('total',
                                                                                             'Million reads')] * 100, 2)
    return df


def collect_metrics(input_file: str) -> pd.DataFrame:
    df = pd.read_csv(input_file, sep='\t', comment='#').T
    complete_df = df.copy()
    df = df.loc[['PF_MISMATCH_RATE', 'PF_INDEL_RATE',
                 'PCT_CHIMERAS', 'MEAN_READ_LENGTH']]
    df.loc['PF_MISMATCH_RATE'] *= 100
    df.loc['PF_INDEL_RATE'] *= 100
    df = pd.DataFrame(df.astype(np.float))
    df = df.round(decimals=2)
    df.index = ['mismatch rate', 'indel rate',
                'chimera rate', 'mean read length']
    return df, complete_df


def generate_comparison_intervals(intervals_file: str, genome_file: str, output_dir: str) -> list:
    bed_files_to_convert = []
    genome_dict_file = '.'.join((splitext(genome_file)[0], 'dict'))
    with open(intervals_file) as chromosomes:
        for chrom in map(lambda x: x.strip().split(), chromosomes):
            with open(pjoin(output_dir, '.'.join(('chr' + chrom[0], 'intervals', 'bed'))), 'w') as outfile:
                bed_files_to_convert.append(outfile.name)
                outfile.write('\t'.join(chrom) + '\n')

    interval_files = [splitext(x)[0] for x in bed_files_to_convert]
    for i in range(len(interval_files)):
        cmd = [
            'picard', 'BedToIntervalList', 'I=%s' % bed_files_to_convert[i],
            'O=%s' % interval_files[i], 'SD=%s' % genome_dict_file]
        with open(pjoin(output_dir, 'logs', 'bed2intv.log'), 'a') as logfile:
            subprocess.check_call(cmd, stdout=logfile, stderr=logfile)

    return interval_files


def generate_header(input_file: str, output_file: str, output_dir: str) -> str:
    output_file = pjoin(output_dir, output_file)
    with open(output_file, 'w') as out:
        cmd = [
            'bcftools', 'view', '-h', input_file]
        with open(pjoin(output_dir, 'logs', 'header.err'), 'w') as errs:
            subprocess.check_call(cmd, stdout=out, stderr=errs)
    return output_file


def mark_duplicates(input_file: list, output_files: list):
    input_bam = input_file[0]
    output_bam, output_metrics, output_log = output_files
    cmd = ['picard', 'MarkDuplicates', 'I=%s' % input_bam, 'M=%s' % output_metrics, 'O=%s' % output_bam,
           'READ_NAME_REGEX=null', 'VALIDATION_STRINGENCY=LENIENT', 'ASO=coordinate']
    with open(output_log, 'w') as out:
        subprocess.check_call(cmd, stdout=out, stderr=out)


def coverage_stats(input_files: list, output_files: list, genome_file: str, intervals: str):
    if intervals is None:
        intervals = input_files[1]
        input_bam = input_files[0][0]
    else:
        input_bam = input_files[0]
    output_metrics, output_log = output_files
    cmd = ['picard', '-Xmx10g', 'CollectWgsMetrics', f"INPUT={input_bam}",
           f"OUTPUT={output_metrics}", f"R={genome_file}", "Q=0",
           'MINIMUM_MAPPING_QUALITY=-1', 'COUNT_UNPAIRED=true',
           'USE_FAST_ALGORITHM=false', 'READ_LENGTH=500', f"INTERVALS={intervals}",
           'VALIDATION_STRINGENCY=LENIENT']
    with open(output_log, 'w') as out:
        out.write(' '.join(cmd) + '\n')
        subprocess.check_call(cmd, stdout=out, stderr=out)


def combine_coverage_metrics(input_files: list, output_file: str, coverage_interval_df: pd.DataFrame) -> None:
    input_files = [x[0] for x in input_files]
    intervals = list(coverage_interval_df['file'])
    classes = list(coverage_interval_df.index)
    convert_dictionary = dict(zip(intervals, classes))

    total_file = [x for x in input_files if splitext(
        basename(intervals[0]))[0] in x][0]
    all_stats, all_histogram = parse_cvg_metrics(total_file)
    all_stats = all_stats.T.loc[
        ['MEAN_COVERAGE', 'MEDIAN_COVERAGE', 'PCT_20X']]
    all_median_coverage = float(all_stats.loc['MEDIAN_COVERAGE', 0])
    class_counts = []
    genome_dfs = []
    for fn in input_files:
        idx = [x for x in intervals if splitext(basename(x))[0] in fn]
        assert len(idx) <= 1, "Non-unique possible source"
        idx = idx[0]

        stats, histogram = parse_cvg_metrics(fn)
        class_counts.append((convert_dictionary[idx], histogram[
                            'high_quality_coverage_count'].sum()))

        ps = np.array(histogram['high_quality_coverage_count'].astype(
            np.float) / histogram['high_quality_coverage_count'].sum())
        distro = np.random.choice(
            np.array(histogram['coverage']) / all_median_coverage, p=ps, size=(1, 20000))
        s = pd.Series(distro[0], name='cvg')
        df = pd.DataFrame(s)
        df['class'] = convert_dictionary[idx]
        genome_dfs.append(df)
    genome_df = pd.concat(genome_dfs)
    class_count = pd.Series(dict(class_counts), name="counts")
    class_count = pd.DataFrame(class_count).reset_index()
    class_count.columns = ['category', 'counts']
    class_count.set_index('category', inplace=True)
    class_count = pd.concat((coverage_interval_df, class_count), axis=1)
    genome_df.to_hdf(output_file, key="coverage_histograms")
    class_count.to_hdf(output_file, key="counts")


def parse_md_file(md_file):
    """Parses mark duplicate Picard output"""
    with open(md_file) as infile:
        out = next(infile)
        while not out.startswith('## METRICS CLASS\tpicard.sam.DuplicationMetrics'):
            out = next(infile)

        res = pd.read_csv(infile, sep='\t')
        return np.round(float(res['PERCENT_DUPLICATION']) * 100, 2)


def parse_cvg_metrics(metric_file):
    """Parses Picard WGScoverage metrics file"""
    with open(metric_file) as infile:
        out = next(infile)
        while not out.startswith('## METRICS CLASS'):
            out = next(infile)

        res1 = pd.read_csv(infile, sep='\t', nrows=1)
    with open(metric_file) as infile:
        out = next(infile)
        while not out.startswith('## HISTOGRAM\tjava.lang.Integer'):
            out = next(infile)

        res2 = pd.read_csv(infile, sep='\t')
    return res1, res2


def generate_rqc_output(dup_ratio: float, metrics: pd.DataFrame, histogram: pd.DataFrame, total_reads: int) -> tuple:
    parameters = metrics.T.loc[['MEAN_COVERAGE', 'MEDIAN_COVERAGE', 'PCT_20X']]
    parameters.loc[('PCT_20X', 0)] = parameters.loc[('PCT_20X', 0)] * 100
    parameters.index = ['mean cvg', 'median cvg', '%>=20x']
    parameters.loc['% duplicated'] = dup_ratio
    parameters.loc['input reads'] = total_reads
    histogram['cum_cov'] = histogram['high_quality_coverage_count'].cumsum(
    ) / histogram['high_quality_coverage_count'].cumsum().max()
    covs = histogram['coverage'].loc[np.searchsorted(
        histogram['cum_cov'], [0.05, 0.1, 0.2, 0.5])]
    covs = np.array(covs.max() / covs).round(2)
    df = pd.DataFrame(data=(covs[:-1]),
                      index=['F95', 'F90', 'F80'], columns=[0])
    parameters = pd.concat((parameters, df))
    parameters = parameters.round(2)
    histogram = histogram.set_index('coverage')
    histogram = pd.DataFrame(
        (histogram['high_quality_coverage_count'] / 1000).round(2))
    histogram.columns = ['loci (x1000)']
    return parameters, histogram


def flatten(lst: list) -> list:
    result = []
    for v in lst:
        if not v:
            continue
        if type(v) == str:
            result.append(v)
        else:
            result = result + flatten(v)

    return result
