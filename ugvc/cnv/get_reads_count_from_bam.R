library(cn.mops)
library(magrittr)

# read arguments
args = commandArgs(trailingOnly=TRUE)
if (length(args)<5) {
  stop("Too few arguments supplied", call.=FALSE)
} else if (length(args)==5) {
  input_bam_file=args[1]
  refSeqNames_string=args[2]
  WL=args[3]
  parallel = args[4]
  base_file_name = args[5]
} else if (length(args)>5) {
  stop("Too many arguments supplied", call.=FALSE)
}

refSeqNames=unlist(strsplit(refSeqNames_string, ","))
bamDataRanges_RC <- getReadCountsFromBAM(input_bam_file, refSeqNames=refSeqNames, WL=strtoi(WL) ,parallel=strtoi(parallel));
save(bamDataRanges_RC, file=paste(base_file_name,".ReadCounts.Rdata",sep = ""));
write.csv(as.data.frame(bamDataRanges_RC),paste(base_file_name,".ReadCounts.csv",sep=""),row.names = FALSE)