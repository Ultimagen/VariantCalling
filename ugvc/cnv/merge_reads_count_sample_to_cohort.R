library(cn.mops)
library(magrittr)

# read arguments
args = commandArgs(trailingOnly=TRUE)
if (length(args)<2) {
  stop("Too few arguments supplied", call.=FALSE)
} else if (length(args)==2) {
  cohort_reads_count_file=args[1]
  sample_reads_count_file=args[2]
} else if (length(args)>2) {
  stop("Too many arguments supplied", call.=FALSE)
}

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
