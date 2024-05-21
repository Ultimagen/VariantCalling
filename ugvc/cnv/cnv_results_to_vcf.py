import pandas as pd
#import numpy as np
import os
from os.path import join as pjoin
#from matplotlib import pyplot as plt
#from matplotlib.patches import Rectangle
import argparse
import logging
import sys
#import seaborn as sns
import pysam
from ugvc import logger
import warnings
warnings.filterwarnings('ignore')


def run(argv):
    """
    converts CNV calls in bed format to vcf. 
    input arguments:
    --cnv_annotated_bed_file: nput bed file holding CNV calls.
    --ref_genome_file: tab delimeted file holding reference genome chr ids with their lengths.
    --out_directory: output directory
    --sample_name: sample name    
    output files:
    vcf file: <sample_name>.cnv.vcf
        shows called CNVs in vcf format. 
    """
    parser = argparse.ArgumentParser(
        prog="cnv_results_to_vcf.py", description="converts CNV calls in bed format to vcf."
    )
    
    parser.add_argument("--cnv_annotated_bed_file", help="input bed file holding CNV calls", required=True, type=str)
    parser.add_argument("--ref_genome_file", help="tab delimeted file holding reference genome chr ids with their lengths", required=True, type=str)
    parser.add_argument("--out_directory", help="output directory", required=False, type=str)
    parser.add_argument("--sample_name", help="sample name", required=True, type=str)
    parser.add_argument("--verbosity",help="Verbosity: ERROR, WARNING, INFO, DEBUG",required=False,default="INFO",)

    args = parser.parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))

    header = pysam.VariantHeader()

    # Add meta-information to the header
    header.add_meta('fileformat', value='VCFv4.2')
    header.add_meta('source', value='ULTIMA_CNV')

    # Add sample names to the header
    sample_name=args.sample_name
    header.add_sample(sample_name)

    header.add_line('##GENOOX_VCF_TYPE=ULTIMA_CNV')

    # Add contigs info to the header
    df_genome = pd.read_csv(args.ref_genome_file,sep='\t',header=None,usecols=[0,1])
    df_genome.columns=['chr','length']
    for index, row in df_genome.iterrows():
       chrID=row['chr']
       length=row['length']
       header.add_line(f"##contig=<ID={chrID},length={length}>")
    
    # Add ALT
    header.add_line('##ALT=<ID=CNV,Description="Copy number variant region">')
    header.add_line('##ALT=<ID=DEL,Description="Deletion relative to the reference">')
    header.add_line('##ALT=<ID=DUP,Description="Region of elevated copy number relative to the reference">')
    
    # Add FILTER
    header.add_line('##FILTER=<ID=PASS,Description="high confidence CNV call">')
    header.add_line('##FILTER=<ID=UG-CNV-LCR,Description="">')
    header.add_line('##FILTER=<ID=LEN,Description="">')

    # Add INFO
    header.add_line('##INFO=<ID=CONFIDENCE,Description="Confidence level for CNV call. 0-Low 1-High">')
    header.add_line('##INFO=<ID=CopyNumber,Description="copy number of CNV call">')
    #header.add_line('##INFO=<ID=END,Description="end position of the CNV">')
    header.add_line('##INFO=<ID=SVLEN,Description="CNV length">')
    header.add_line('##INFO=<ID=SVTYPE,Description="CNV type. can be DUP or DEL">')
    
    # Add FORMAT
    header.add_line('##FORMAT=<ID=GT,Description="Genotype">')

    # Open a VCF file for writing
    if args.out_directory: 
        out_directory=args.out_directory
    else:
        out_directory=""
    outfile=pjoin(out_directory,sample_name+'.cnv.vcf')
    
    with pysam.VariantFile(outfile, mode='w', header=header) as vcf_out:
        df_cnvs = pd.read_csv(args.cnv_annotated_bed_file,sep="\t",header=None)
        df_cnvs.columns=['chr','start','end','info']
        for index, row in df_cnvs.iterrows():
            # Create a new VCF record
            chrID = row['chr']
            start = row['start']
            end = row['end']
            info = row['info']

            CN = int(info.split("|")[0].replace("CN",""))
            cnv_type = 'DUP'
            if CN<2:
                cnv_type='DEL'

            filters = []
            for item in info.split(';'):
                arr = item.split('|')
                if len(arr)>1:
                    filters.append(arr[1])
            
            record = vcf_out.new_record()
            record.contig =  chrID
            record.start = start
            record.stop = end
            record.ref = "N"
            record.alts = (cnv_type,)

            CONFIDENCE = 'HIGH'
            if len(filters)>0 :
                for f in filters:
                    record.filter.add(f)
                CONFIDENCE='MEDIUM'
                if len(filters)>1 :
                    CONFIDENCE='LOW'
            else:
                record.filter.add('PASS')
          
            record.info['CONFIDENCE'] = CONFIDENCE
            record.info['CopyNumber'] = str(CN)
            record.info['SVLEN'] = str(int(end)-int(start))
            record.info['SVTYPE'] = cnv_type


            # Set genotype information for each sample
            GT=[None,1]
            if CN==1:
                GT=[0,1]
            elif CN==0:
                GT=[1,1]
            record.samples[sample_name]['GT'] = (GT[0],GT[1]) 

            # Write the record to the VCF file
            vcf_out.write(record)
        
        vcf_out.close()
        logger.info(f"output file: {outfile}")

if __name__ == "__main__":
    run(sys.argv)