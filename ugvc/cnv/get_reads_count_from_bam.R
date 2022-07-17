library(cn.mops)
library(magrittr)
library(argparse)

# create parser object
parser <- ArgumentParser()

parser$add_argument("-i", "--input_bam_file",
                    help="input bam file path")
parser$add_argument("-refseq", "--refSeqNames_string",
                    help="chromosomes names in comma seperated format. e.g, chr1,chr2,chrX" ,
                    default="chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX,chrY")
parser$add_argument("-wl", "--window_length",
                   help="window length (#bp) for which reads count is calculated for",
                   type="integer", default=1000)
parser$add_argument("-p", "--parallel",
                    help="number of parallel process",
                    type="integer", default=30)
parser$add_argument("-o", "--base_file_name",
                    help="out base file name")

# read arguments
# args = commandArgs(trailingOnly=TRUE)
# if (length(args)<5) {
#   stop("Too few arguments supplied", call.=FALSE)
# } else if (length(args)==5) {
#   input_bam_file=args[1]
#   refSeqNames_string=args[2]
#   WL=args[3]
#   parallel = args[4]
#   base_file_name = args[5]
# } else if (length(args)>5) {
#   stop("Too many arguments supplied", call.=FALSE)
# }
args <- parser$parse_args()

refSeqNames=unlist(strsplit(args$refSeqNames_string, ","))
bamDataRanges_RC <- getReadCountsFromBAM(args$input_bam_file, refSeqNames=args$refSeqNames, WL=args$window_length ,parallel=args$parallel);
save(bamDataRanges_RC, file=paste(args$base_file_name,".ReadCounts.Rdata",sep = ""));
write.csv(as.data.frame(bamDataRanges_RC),paste(args$base_file_name,".ReadCounts.csv",sep=""),row.names = FALSE)