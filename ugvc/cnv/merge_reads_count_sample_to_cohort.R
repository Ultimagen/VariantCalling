suppressPackageStartupMessages(library(cn.mops))
suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(rhdf5))

parser <- ArgumentParser()
parser$add_argument("-cohort_rc", "--cohort_reads_count_file",
                    help="input cohort reads count file in rds format")
parser$add_argument("-sample_rc", "--sample_reads_count_file",
                    help="input sample reads count file in rds format")
args <- parser$parse_args()

cohort_reads_count_file <- args$cohort_reads_count_file
sample_reads_count_file <- args$sample_reads_count_file
# load reads count rds files
gr1<- readRDS(file = cohort_reads_count_file )
gr2<-readRDS(file = sample_reads_count_file)

# merge sample to cohort
merged_cohort_reads_count <- GRanges(
  seqnames = seqnames(gr1),
  ranges = ranges(gr1),
  strand = strand(gr1),
  mcols(gr1),
  mcols(gr2)
)

# save merged cohort
saveRDS(merged_cohort_reads_count,file="merged_cohort_reads_count.rds")
h5write(as.data.frame(merged_cohort_reads_count),"merged_cohort_reads_count.hdf5","merged_cohort_reads_count")

