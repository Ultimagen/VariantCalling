# Functions that for variant calling pipeline
import subprocess
import configargparse

from os.path import join as pjoin
from os.path import basename, dirname, abspath
import os
import sys
dname = dirname(abspath(__file__))
sys.path.append(pjoin(dname, ".."))
import utils
import pandas as pd
# Align and merge
def parse_params_file(params_file):
    ap = configargparse.ArgParser()
    ap.add("-c", required=True, is_config_file=True, help='config file path')
    ap.add('--em_vc_demux_file', required=True)
    ap.add('--em_vc_genome', required=True, help="Path to genome file (bwa index, dict should exist)")
    ap.add('--em_vc_output_dir', required=True, help="Output directory")
    ap.add('--em_vc_chromosomes_list', required=False, help="File with the list of chromosomes to test")
    ap.add('--em_vc_recalibration_model', required=False, help="recalibration model (h5)")
    ap.add('--em_vc_number_to_sample', required=False, help="Number of records to downsample", type=int)
    ap.add('--em_vc_number_of_cpus', required=False, help="Number of CPUs on the machine", type=int, default = 12)
    return ap.parse_known_args()[0]

def head_file( input_file, output_file, number_to_sample, nthreads) : 
    output_bam, output_err = output_file 
    cmd1 = ["samtools", "view", '-@%d'%int((nthreads-2)/2), '-h', input_file]
    cmd2 = ["head", f"-{number_to_sample+100}"]
    cmd3 = ["samtools", "view", '-@%d'%int((nthreads-2)/2), "-b", "-o", output_bam]
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
    nthreads_alignment = int(0.8*(nthreads-1))
    nthreads_samtools= int(0.2*(nthreads-1))
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

def align_and_merge( input_file, output_file, genome_file, nthreads ): 


    output_bam, output_err = output_file 
    output_err_handle = open(output_err, "w")    

    nthreads_alignment = int(nthreads-3)
    

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

# Prepare fetch intervals 
def prepare_fetch_intervals( input_file, output_file, genome_file ) : 
    genome_dct = utils.get_chr_sizes(genome_file + ".sizes")
    with open(output_file,'w') as output_handle: 
        with open(input_file) as input_handle: 
            for chrom in input_handle : 
                output_handle.write(f"{chrom.strip()}\t0\t{genome_dct[chrom.strip()]}\n")


def filter_quality( input_file, output_files, nthreads ):
    output_bam, output_err = output_files
    input_file = input_file[0]
    cmd = [ 'samtools', 'view', '-q20', '-@%d'%(nthreads-1), '-b', '-o',output_bam, input_file]
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
        cmd1 = [ 'samtools', 'sort', '-@%d'%(nthreads-1), '-T', dirname(output_bam), '-o' , output_bam, input_file[0] ]
        subprocess.check_call(cmd1, stderr=output_err_handle)

def recalibrate_file( input_file, output_files, recalibration_model,nthreads ): 
    input_file = input_file[0]
    output_bam, output_err = output_files
    pthreads = int(0.8 * nthreads)
    dthreads = int(0.1 * nthreads)
    cthreads = int(0.1 * nthreads)
    with open(output_err,'w') as output_err_handle : 
        cmd1 = ["LD_LIBRARY_PATH=/usr/local/lib", 
         "/home/ubuntu/proj/Utils/recalibration/recalibrate", 
         "--input=%s"%input_file, "--binary",
         "--output=%s"%output_bam, "--model=%s"%recalibration_model,
         "--threshold=0.003", "--dthreads=%d"%dthreads, "--pthreads=%d"%pthreads, "--cthreads=%d"%cthreads]
        subprocess.check_call(" ".join(cmd1), stdout=output_err_handle, stderr=output_err_handle, shell=True)



def index_file( input_file, output_file ): 
    output_bam, output_err = output_file 
    with open(output_err, "w") as output_err_handle  : 
        cmd1 = [ 'samtools', 'index', input_file[0] ]
        subprocess.check_call(cmd1, stderr=output_err_handle)


def create_variant_calling_intervals( input_file, output_file ): 
    output_bed, output_err = output_file 
    with open(output_bed,'w') as output_bed_handle : 
        with open(output_err, 'w') as output_err_handle : 
            cmd1 = [ 'bedtools', 'makewindows', '-b', input_file, '-n','10' ]
            subprocess.check_call(cmd1, stdout=output_bed_handle, stderr=output_err_handle)


def split_intervals_into_files( input_file, output_files, output_file_name_root ) : 
    for oo in output_files : 
        os.unlink(oo)

    with open(input_file[0]) as input_file_handle : 
        for i,line in enumerate(input_file_handle ) : 
            with open(f"{output_file_name_root}.{i+1}",'w') as out : 
                lsp = line.split()
                out.write(f"%s:%d-%d\n"%(lsp[0], int(lsp[1])+1, int(lsp[2])))

def variant_calling (input_files, output_files, genome_file) : 
    aligned_bam, interval_file = input_files
    aligned_bam = aligned_bam[0]
    output_vcf, output_log = output_files 

    gatk="/home/ubuntu/software/gatk/gatk"
    my_env = os.environ.copy()
    my_env["GATK_LOCAL_JAR"] = "/home/ubuntu/software/gatk/gatk-package-ultima-v0.2-6-ge3bfd5d-SNAPSHOT-local.jar"

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

def collect_idxstats( input_file: str ) -> pd.DataFrame : 
    df = pd.read_csv(input_file, sep="\t", 
            header=None, index_col=0, names=['length','aligned_reads', 'unaligned_reads'])
    df = df.sum()
    df.drop(['length'], inplace=True)
    return df

def collect_metrics( input_file: str) -> pd.DataFrame : 
    df = pd.read_csv(input_file, sep="\t",comment="#").T
    return df.loc[['PF_MISMATCH_RATE', 'PF_INDEL_RATE', 'PCT_CHIMERAS']]


