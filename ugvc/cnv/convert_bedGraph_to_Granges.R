suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library("GenomicRanges"))

parser <- ArgumentParser()

parser$add_argument("-i", "--input_bedGraph",
                    help="input bedGraph file path")
args <- parser$parse_args()

bg <- read.table(args$input_bedGraph, sep="\t", skip=1)
colnames(bg) <- c("seqnames", "start", "end", "value")
gr <- makeGRangesFromDataFrame(bg, ignore.strand=TRUE, keep.extra.columns=TRUE)
out_file_name <- paste(tools::file_path_sans_ext(args$input_bedGraph), ".ReadCounts.rds", sep="")
saveRDS(gr, file=out_file_name)
