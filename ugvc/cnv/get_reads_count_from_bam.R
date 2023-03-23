suppressPackageStartupMessages(library(cn.mops))
suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(rhdf5))


parser <- ArgumentParser()

parser$add_argument("-i", "--input_bam_file",
                    help="input bam file path")
parser$add_argument("-refseq", "--refSeqNames_string",
                    help="chromosome names in comma seperated format. e.g, chr1,chr2,chrX" ,
                    default="chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX,chrY")
parser$add_argument("-wl", "--window_length",
                   help="window length (#bp) for which reads count is calculated for",
                   type="integer", default=1000)
parser$add_argument("-p", "--parallel",
                    help="number of parallel processes",
                    type="integer", default=30)
parser$add_argument("-o", "--base_file_name",
                    help="out base file name")
parser$add_argument("--save_hdf", action='store_true',
                    help="whether to save reads count data-frames in hdf5 format")

args <- parser$parse_args()

refSeqNames <- unlist(strsplit(args$refSeqNames_string, ","))
bamDataRanges_RC <- getReadCountsFromBAM(args$input_bam_file, refSeqNames=refSeqNames, WL=args$window_length ,parallel=args$parallel)
saveRDS(bamDataRanges_RC, file = paste(args$base_file_name,".ReadCounts.rds",sep = ""))

if(args$save_hdf){
  hdf5_out_file_name <- paste(args$base_file_name,".ReadCounts.hdf5",sep = "")
  h5createFile(hdf5_out_file_name)
  h5write(as.data.frame(bamDataRanges_RC), hdf5_out_file_name,"bamDataRanges_RC")
}