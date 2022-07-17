suppressPackageStartupMessages(library(cn.mops))
suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(argparse))

parser <- ArgumentParser()
parser$add_argument("-cohort_rc", "--cohort_reads_count_file",
                    help="input cohort reads count file in Rdata format")
parser$add_argument("-sample_rc", "--sample_reads_count_file",
                    help="input sample reads count file in Rdata format")
args <- parser$parse_args()

cohort_reads_count_file <- args$cohort_reads_count_file
sample_reads_count_file <- args$sample_reads_count_file

# load reads count Rdata file
RC_obj<-load(cohort_reads_count_file)
gr1<-get(RC_obj)
rm(RC_obj)

RC_obj<-load(sample_reads_count_file)
gr2<-get(RC_obj)
rm(RC_obj)

# merge sample to cohort
merged_cohort_reads_count <- GRanges(
  seqnames = seqnames(gr1),
  ranges = ranges(gr1),
  strand = strand(gr1),
  mcols(gr1),
  mcols(gr2)
)

# save merged cohort
save(merged_cohort_reads_count,file="merged_cohort_reads_count.RData")
write.csv(as.data.frame(merged_cohort_reads_count),"merged_cohort_reads_count.csv", row.names = FALSE)
