suppressPackageStartupMessages(library("argparse"))
suppressPackageStartupMessages(library("GenomicRanges"))

parser <- ArgumentParser()

parser$add_argument("-i", "--input_bedGraph",
                    help="input bedGraph file path")
parser$add_argument("-sample_name", "--sample_name",
                    help="input bedGraph file path")

args <- parser$parse_args()


bg <- read.table(args$input_bedGraph, sep="\t", skip=1)
colnames(bg) <- c("seqnames", "start", "end", args$sample_name)
gr <- makeGRangesFromDataFrame(bg, ignore.strand=TRUE, keep.extra.columns=TRUE)
out_file_name <- paste(args$sample_name, ".ReadCounts.rds", sep="")
saveRDS(gr, file=out_file_name)
