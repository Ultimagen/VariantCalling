# Functions that for variant calling pipeline
import subprocess
import configargparse

from os.path import join as pjoin
from os.path import basename, dirname, abspath, splitext
import os
import sys
dname = dirname(abspath(__file__))
sys.path.append(pjoin(dname, ".."))
import utils
import pandas as pd
import numpy as np
import pysam

def parse_params_file(params_file, pipeline_name):
    ap = configargparse.ArgParser()
    ap.add("-c", required=True, is_config_file=True, help='config file path')
    group1 = ap.add_mutually_exclusive_group(required=True)
    group1.add('--DataFileName', help="Path + prefix of the output_files")
    group1.add('--em_vc_output_dir', help="Output dir")
    ap.add('--em_vc_genome', required=True, help="Path to genome file (bwa index, dict should exist)")
    ap.add('--em_vc_chromosomes_list', required=False, help="File with the list of chromosomes to test")
    ap.add('--em_vc_number_of_cpus', required=False, help="Number of CPUs on the machine", type=int, default = 12)
    if pipeline_name == "error_metrics":
        ap.add('--em_vc_demux_file', help="Path to the demultiplexed bam")
        ap.add('--em_vc_number_to_sample', required=False, help="Number of records to downsample", type=int)
    elif pipeline_name == "rapidqc" : 
        ap.add('--rqc_demux_file', help="Path to the demultiplexed bam")

        ap.add('--rqc_evaluation_intervals', required=False, help="file that contains a list of Intervals to evaluate on (interval_list of picard)", type=str)
    elif pipeline_name == "variant_calling":
        ap.add('--em_vc_demux_file', help="Path to the demultiplexed bam")
        ap.add('--em_vc_recalibration_model', required=False, help="recalibration model (h5)")    
        ap.add('--em_vc_ground_truth', required=False, help="Ground truth file to compare", type=str)
        ap.add('--em_vc_ground_truth_highconf', required=False, help="Ground truth high confidence file", type=str)    
        ap.add('--em_vc_gaps_hmers_filter', required=False, help="Bed file with regions to filter out", type=str)
    else: 
        raise RuntimeError(f"{pipeline_name} is not a defined pipeline")

    args = ap.parse_known_args()[0]

    if pipeline_name == 'rapidqc': 
        # We need both the parameters em_vc_demux_file and fqc_demux_file
        # to live in the same configuration file that runs two pipelines 
        # and point to two different files
        # this is why I am using fqc_demux_file in the parameter list and not 
        # em_vc_demux_file for both. 
        args.em_vc_demux_file = args.rqc_demux_file
        args.em_vc_basename = basename(args.rqc_demux_file)

    if args.DataFileName is not None:
        args.em_vc_output_dir = dirname(args.DataFileName)
        args.em_vc_basename = basename(args.DataFileName)
    else :
        args.em_vc_basename = basename(args.em_vc_demux_file)
    
    return args

def head_file( input_file, output_file, number_to_sample, nthreads) : 
    output_bam, output_err = output_file 
    cmd1 = ["samtools", "view", '-@%d'%max(1,int((nthreads-2)/2)), '-h', input_file]
    cmd2 = ["head", f"-{number_to_sample+100}"]
    cmd3 = ["samtools", "view", '-@%d'%max(1,int((nthreads-2)/2)), "-b", "-o", output_bam]
    with open(output_err, 'w') as output_err_handle : 
        task1 = subprocess.Popen(cmd1, stdout = subprocess.PIPE, stderr=output_err_handle)
        task2 = subprocess.Popen(cmd2, stdin = task1.stdout, stdout = subprocess.PIPE, stderr=output_err_handle)
        task1.stdout.close()
        task3 = subprocess.Popen(cmd3, stdin = task2.stdout, stderr=output_err_handle)
        task2.stdout.close()
        output=task3.communicate()

def align( input_file, output_file, genome_file, nthreads ) : 
    output_bam, output_err = output_file 
    output_err_handle = open(output_err, "w")
    nthreads_alignment = max(1,int(0.8*(nthreads-1)))
    nthreads_samtools= max(1,int(0.2*(nthreads-1)))
    if type(input_file) == list:
        cmd1 = ['picard' ,'-Xms5000m','SamToFastq', 'INPUT=%s'%input_file[0], 'FASTQ=/dev/stdout']
    else : 
        cmd1 = ['picard' ,'-Xms5000m','SamToFastq', 'INPUT=%s'%input_file, 'FASTQ=/dev/stdout']

    output_err_handle.write(" ".join(cmd1))
    output_err_handle.flush()
    task1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=output_err_handle)
    cmd2 = ['bwa','mem', '-t', str(nthreads_alignment), genome_file, '-']
    output_err_handle.write(" | ")
    output_err_handle.write(" ".join(cmd2))
    output_err_handle.flush()
    task2 = subprocess.Popen(cmd2, stdin = task1.stdout, stdout = subprocess.PIPE, stderr = output_err_handle)
    task1.stdout.close()
    cmd3 = ["samtools", 'view', '-@%d'%nthreads_samtools, '-b', '-o', output_bam, '-']
    output_err_handle.write(" | ")
    output_err_handle.write(" ".join(cmd2))
    output_err_handle.flush()    
    task3 = subprocess.Popen(cmd3, stdin=task2.stdout, stdout=output_err_handle, stderr=output_err_handle)
    task2.stdout.close()
    output = task3.communicate()
    output_err_handle.close()
    if task3.returncode!=0: 
        raise RuntimeError("Alignment failed")

    

def align_and_merge( input_file, output_file, genome_file, nthreads ): 


    output_bam, output_err = output_file 
    output_err_handle = open(output_err, "w")    

    nthreads_alignment = max(1,int(nthreads-3))
    

    cmd1 = ['picard' ,'-Xms5000m','SamToFastq', 'INPUT=%s'%input_file, 'FASTQ=/dev/stdout']
    output_err_handle.write(" ".join(cmd1))
    output_err_handle.flush()
    task1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=output_err_handle)
    cmd2 = ['bwa','mem', '-t', str(nthreads_alignment), genome_file, '-']
    output_err_handle.write(" | ")
    output_err_handle.write(" ".join(cmd2))
    output_err_handle.flush()
    
    task2 = subprocess.Popen(cmd2, stdin = task1.stdout, stdout = subprocess.PIPE, stderr = output_err_handle)
    task1.stdout.close()

    cmd3 = ["picard", "-Xms3000m", "MergeBamAlignment", "VALIDATION_STRINGENCY=SILENT",
        "ATTRIBUTES_TO_RETAIN=X0", "ATTRIBUTES_TO_REMOVE=NM", "ATTRIBUTES_TO_REMOVE=MD",
        "ALIGNED_BAM=/dev/stdin", "UNMAPPED_BAM=%s"%input_file, "OUTPUT=%s"%output_bam, 
        "REFERENCE_SEQUENCE=%s"%genome_file, "PAIRED_RUN=false", 'SORT_ORDER="unsorted"', 
        "IS_BISULFITE_SEQUENCE=false", "ALIGNED_READS_ONLY=true", "CLIP_ADAPTERS=false", 
        "MAX_RECORDS_IN_RAM=2000000", "MAX_INSERTIONS_OR_DELETIONS=-1", "PRIMARY_ALIGNMENT_STRATEGY=MostDistant",
        "UNMAP_CONTAMINANT_READS=true", "ADD_PG_TAG_TO_READS=false"]
    output_err_handle.write(" | ")
    output_err_handle.write(" ".join(cmd2))
    output_err_handle.flush()
    
    task3 = subprocess.Popen(cmd3, stdin=task2.stdout, stdout=output_err_handle, stderr=output_err_handle)
    task2.stdout.close()
    output = task3.communicate()
    output_err_handle.close()
    if task3.returncode!=0 : 
        raise RuntimeError("Alignment failed")

def align_minimap_and_filter( input_file, output_files, genome_file, nthreads, chromosome_file) : 

    the_chromosome = [x.strip() for x in open(chromosome_file) if x ]
    assert len(the_chromosome) == 1, "Chromosome file should contain a single chromosome"
    the_chromosome = the_chromosome[0]
    output_bam, output_err = output_files
    input_file = input_file

    alntmp = pysam.AlignmentFile(input_file, check_sq=False)
    header = str(alntmp.header).split("\n")
    rg_line = [ x for x in header if x.startswith("@RG")]
    assert len(rg_line)==1, "uBAM does not contain RG or it is not single"
    rg_line = rg_line[0]
    rg_line = rg_line.replace("\t","\\t")
    alntmp.close()

    with open(output_err,'w') as outlog : 

        minimap_threads = max(1, int(0.8*nthreads))
        samtools_in_threads = max(1, int(0.2*nthreads))
        cmd1 = ['samtools', 'fastq','-@', str(samtools_in_threads), '-t', input_file]
        cmd2 = ['minimap2', '-t', str(minimap_threads), '-R', rg_line, '-x', 'sr', '-y', '-a', genome_file,'-']
        cmd3 = ['awk', f'($3=="{the_chromosome}") || ($1 ~ /^@/)']
        cmd4 = ['samtools', 'view', '-b', '-o', output_bam,'-' ]
        outlog.write('|'.join((" ".join(cmd1), " ".join(cmd2), " ".join(cmd3), " ".join(cmd4)))+"\n")
        outlog.flush()
        task1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr = outlog)
        task2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr = outlog, stdin=task1.stdout)
        task1.stdout.close()
        task3 = subprocess.Popen(cmd3, stdout=subprocess.PIPE, stderr = outlog, stdin=task2.stdout)        
        task2.stdout.close()
        task4 = subprocess.Popen(cmd4, stdout=outlog, stderr = outlog, stdin=task3.stdout)        
        task3.stdout.close()
        output = task4.communicate()
        rcs = [ x.returncode for x in [task1, task2, task3, task4]]
        for i in range(len(rcs)):
            result = ""
            if rcs[i]!=0:
                result += f"Task{i},"
        if len(result) > 0  :
            raise RuntimeError(f"{result} of alignment failed")

#Prepare fetch intervals 
def prepare_fetch_intervals( input_file, output_file, genome_file ) : 
    genome_dct = utils.get_chr_sizes(genome_file + ".sizes")
    with open(output_file,'w') as output_handle: 
        with open(input_file) as input_handle: 
            for chrom in input_handle : 
                if chrom :
                    output_handle.write(f"{chrom.strip()}\t0\t{genome_dct[chrom.strip()]}\n")


def filter_quality( input_file, output_files, nthreads ):
    output_bam, output_err = output_files
    input_file = input_file[0]
    cmd = [ 'samtools', 'view', '-q20', '-@%d'%max(1,(nthreads-1)), '-b', '-o',output_bam, input_file]
    with open(output_err,'w') as output_err_handle : 
        subprocess.check_call(cmd, stdout=output_err_handle, stderr=output_err_handle)

def fetch_intervals( input_file, output_files): 
    output_bam, output_err = output_files
    filter_intervals = input_file[1]
    input_file = input_file[0][0]
    cmd1 = ['bedtools','intersect','-wa','-a', input_file, '-b', filter_intervals ]
    output_err_handle = open(output_err,'w')
    output_file_handle = open(output_bam,'w')
    subprocess.check_call(cmd1, stdout=output_file_handle, stderr=output_err_handle)
    output_file_handle.close()
    output_err_handle.close()

def sort_file ( input_file, output_file, nthreads ): 
    output_bam, output_err = output_file 
    with open(output_err, "w") as output_err_handle  : 
        cmd1 = [ 'samtools', 'sort', '-@%d'%max(1,(nthreads-1)), '-T', dirname(output_bam), '-o' , output_bam, input_file[0] ]
        subprocess.check_call(cmd1, stderr=output_err_handle)

def recalibrate_file( input_file, output_files, recalibration_model,nthreads ): 
    input_file = input_file[0]
    output_bam, output_err = output_files
    pthreads = max(1,int(0.8 * nthreads))
    dthreads = max(1,int(0.1 * nthreads))
    cthreads = max(1,int(0.1 * nthreads))
    with open(output_err,'w') as output_err_handle : 
        cmd1 = ["LD_LIBRARY_PATH=/usr/local/lib", 
         "recalibrate", 
         "--input=%s"%input_file, "--binary",
         "--output=%s"%output_bam, "--model=%s"%recalibration_model,
         "--threshold=0.003", "--dthreads=%d"%dthreads, "--pthreads=%d"%pthreads, "--cthreads=%d"%cthreads]
        subprocess.check_call(" ".join(cmd1), stdout=output_err_handle, stderr=output_err_handle, shell=True)



def index_file( input_file, output_file ): 
    output_bam, output_err = output_file 
    with open(output_err, "w") as output_err_handle  : 
        cmd1 = [ 'samtools', 'index', input_file[0] ]
        subprocess.check_call(cmd1, stderr=output_err_handle)


def create_variant_calling_intervals( input_file, output_file, n_threads ): 
    output_bed, output_err = output_file 
    with open(output_bed,'w') as output_bed_handle : 
        with open(output_err, 'w') as output_err_handle : 
            cmd1 = [ 'bedtools', 'makewindows', '-b', input_file, '-n',str(n_threads) ]
            subprocess.check_call(cmd1, stdout=output_bed_handle, stderr=output_err_handle)


def split_intervals_into_files( input_file, output_files, output_file_name_root ) : 
    for oo in output_files : 
        os.unlink(oo)

    with open(input_file[0]) as input_file_handle : 
        for i,line in enumerate(input_file_handle ) : 
            with open(f"{output_file_name_root}.{i+1}",'w') as out : 
                lsp = line.split()
                out.write(f"%s:%d-%d\n"%(lsp[0], int(lsp[1])+1, int(lsp[2])))

def intersect_intervals( input_files: list, output_files: list ) :
    left, right=input_files
    output, log = output_files 
    with open(log,'w') as out : 
        cmd = ['picard','IntervalListTools',f'I={left},{right}', 
        'ACTION=INTERSECT',f'O={output}']
        out.write(" ".join(cmd)+"\n")
        out.flush()
        subprocess.check_call(cmd, stdout=out, stderr=out)

def variant_calling (input_files, output_files, genome_file) : 
    aligned_bam, interval_file = input_files
    aligned_bam = aligned_bam[0]
    output_vcf, output_log = output_files 

    gatk="gatk"
    my_env = os.environ.copy()

    interval = [ x.strip() for x in open(interval_file) ][0]
    
    cmd1 = [gatk, "HaplotypeCaller", "-I", aligned_bam, "-O", output_vcf, 
    "-R", genome_file, "--intervals", interval ,"--likelihood-calculation-engine", "FlowBased"]
    with open(output_log,'w') as output_log_handle : 
        output_log_handle.write(" ".join(cmd1))
        output_log_handle.write("\n")
        task = subprocess.Popen(cmd1, env=my_env, stdout=output_log_handle, stderr=output_log_handle)
        task.wait()
    if task.returncode!=0:
        raise RuntimeError(" ".join(cmd1) + " exited abnormally")

def error_metrics( input_files, output_files, genome_file) : 
    aligned_bam, _ = input_files
    output_metrics, output_log = output_files 
    cmd = ["picard", "CollectAlignmentSummaryMetrics", "R=%s"%genome_file, 
     "I=%s"%aligned_bam, "O=%s"%output_metrics, "VALIDATION_STRINGENCY=LENIENT"]
    
    with open(output_log,'w') as output_err_handle:
        subprocess.check_call(cmd, stdout=output_err_handle, stderr=output_err_handle)


def idxstats( input_files, output_files ) : 
    aligned_bam, _ = input_files
    output_stats, output_log = output_files 
    cmd = ["samtools", "idxstats", aligned_bam]
    
    with open(output_log,'w') as output_err_handle:
        with open(output_stats,'w') as output_stats_handle : 
            subprocess.check_call(cmd, stdout=output_stats_handle, stderr=output_err_handle)

def collect_alnstats( idxstats_file: str, filter_metrics: str ) -> pd.DataFrame : 
    '''
    Parameters
    ----------
    idxstats_file: str
        idxstats output
    filter_metrics: str
        Alignment metrics filter
    '''
    df = pd.read_csv(idxstats_file, sep="\t", engine="python",
            header=None, index_col=0, names=['length','aligned_reads', 'unaligned_reads'])
    df = df.sum()
    df.drop(['length'], inplace=True)
    df1 = pd.read_csv(filter_metrics, sep="\t",comment="#", engine="python").T[0]
    df['hq_aligned_reads'] = df1.loc['TOTAL_READS']
    df['total_reads'] = df['aligned_reads'] + df['unaligned_reads']
    #df['pct_aligned'] = df['aligned_reads']/ (df['unaligned_reads'] + df['aligned_reads'])*100
    #df['pct_high_quality'] = df['hq_aligned_reads']/ df['aligned_reads']*100     

    df = df.loc[['total_reads','aligned_reads', 'hq_aligned_reads','unaligned_reads']]
    df.index = ['total','aligned', 'hq aligned','unaligned']
    df = pd.DataFrame(df)
    df.columns = ['Million reads']
    df['Million reads'] = (df['Million reads']/1e6).round(decimals=2)
    df['%'] = 100

    df.loc['aligned','%'] = np.round(df.loc['aligned','Million reads']/df.loc['total','Million reads']*100,2)
    df.loc['unaligned','%'] = np.round(df.loc['unaligned','Million reads']/df.loc['total','Million reads']*100,2)
    df.loc['hq aligned','%'] = np.round(df.loc['hq aligned','Million reads']/df.loc['total','Million reads']*100,2)

    return df

def collect_metrics( input_file: str) -> pd.DataFrame : 
    df = pd.read_csv(input_file, sep="\t",comment="#").T
    df = df.loc[['PF_MISMATCH_RATE', 'PF_INDEL_RATE', 'PCT_CHIMERAS']]
    df.loc['PF_MISMATCH_RATE']*=100
    df.loc['PF_INDEL_RATE']*=100
    df = pd.DataFrame(df.astype(np.float))
    df = df.round(decimals=2)
    df.index = ['mismatch rate', 'indel rate', 'chimera rate']
    
    return df

def generate_comparison_intervals( intervals_file: str, genome_file: str, output_dir: str ) -> list: 

    bed_files_to_convert = []
    genome_dict_file = '.'.join((splitext(genome_file)[0],'dict'))
    with open(intervals_file) as chromosomes : 
        for chrom in map(lambda x: x.strip().split(), chromosomes) : 
            with open(pjoin(output_dir, ".".join(("chr"+chrom[0], 'intervals','bed'))),'w') as outfile : 
                bed_files_to_convert.append(outfile.name)
                outfile.write("\t".join(chrom) + "\n")
    interval_files = [ splitext(x)[0] for x in bed_files_to_convert]
    for i in range(len(interval_files)) : 
        cmd = ['picard', 'BedToIntervalList','I=%s'%bed_files_to_convert[i], 
        'O=%s'%interval_files[i], 'SD=%s'%genome_dict_file]
        with open(pjoin(output_dir, "logs", "bed2intv.log"),'a') as logfile : 
            subprocess.check_call(cmd, stdout=logfile, stderr=logfile)
    return interval_files 

def generate_header( input_file: str, output_file: str, output_dir: str) -> str: 
    output_file = pjoin(output_dir, output_file)
    with (open(output_file, 'w')) as out : 
        cmd = ['bcftools', 'view', '-h', input_file]
        with open(pjoin(output_dir, 'logs', "header.err"),'w') as errs : 
            subprocess.check_call(cmd, stdout = out, stderr = errs )
    return output_file


def mark_duplicates( input_file: list, output_files: list ) : 
    input_bam = input_file[0]
    output_bam, output_metrics, output_log = output_files
    cmd = ['picard', 'MarkDuplicates', 'I=%s'%input_bam, 'M=%s'%output_metrics, 'O=%s'%output_bam, 
    'READ_NAME_REGEX=null', 'VALIDATION_STRINGENCY=LENIENT','ASO=coordinate']
    with open(output_log,'w') as out :  
        subprocess.check_call(cmd, stdout=out, stderr=out)

def coverage_stats( input_files: list, output_files: list, genome_file: str) : 
    input_bam, name, intervals = input_files
    output_metrics, output_log = output_files
    cmd = ['picard', '-Xmx10g', 'CollectWgsMetrics',f'INPUT={input_bam}', 
    f'OUTPUT={output_metrics}', f'R={genome_file}', 
    'MINIMUM_MAPPING_QUALITY=-1','COUNT_UNPAIRED=true', 
    'USE_FAST_ALGORITHM=false', 'READ_LENGTH=500', f'INTERVALS={intervals}', 
    'VALIDATION_STRINGENCY=LENIENT']
    with open(output_log,'w') as out : 
        out.write(" ".join(cmd)+"\n")
        subprocess.check_call(cmd, stdout=out, stderr=out)

# def gc_bias( input_file: list, output_files: list, genome_file: str) : 
#     input_bam = input_file[0]
#     output_metrics, output_log = output_files 
#     cmd = ['picard', '-Xmx10g', 'CollectGcBiasMetrics', 
#     f'INPUT={input_bam}', f'OUTPUT={output_metrics}', 
#     f'SUMMARY_METRICS={output_summary_metrics}', f'REFERENCE_SEQUENCE={genome_file}']
#     with open(output_log,'w') as out : 
#         out.write(" ".join(cmd)+"\n")
#         subprocess.check_call(cmd, stdout=out, stderr=out)
