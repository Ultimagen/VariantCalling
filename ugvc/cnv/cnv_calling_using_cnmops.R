library(cn.mops)
library(magrittr)

# read arguments
args = commandArgs(trailingOnly=TRUE)
if (length(args)<2) {
  stop("Too few arguments supplied", call.=FALSE)
} else if (length(args)==2) {
  cohort_reads_count_file=args[1]
  minWidth_val = args[2]
} else if (length(args)>2) {
  stop("Too many arguments supplied", call.=FALSE)
}

load(cohort_reads_count_file) #variable name : merged_cohort_reads_count
cores=30

resCNMOPS <- cn.mops(merged_cohort_reads_count,parallel=cores,minWidth = as.integer(minWidth_val) )
resCNMOPS_Int <-calcIntegerCopyNumbers(resCNMOPS)

save(resCNMOPS_Int,file="cohort.cnmops.RData")

df_cnvs<-as.data.frame(cnvs(resCNMOPS_Int))
write.csv(df_cnvs,"cohort.cnmops.cnvs.csv", row.names = FALSE,quote=FALSE)

df_cnvr<-as.data.frame(cnvr(resCNMOPS_Int))
write.csv(df_cnvr,"cohort.cnmops.cnvr.csv", row.names = FALSE,quote=FALSE)

individualCall_df<-as.data.frame(individualCall(resCNMOPS_Int))
write.csv(individualCall_df,"cohort.cnmops.individualCall.csv",row.names = FALSE, quote=FALSE)

inicall_df<-as.data.frame(iniCall(resCNMOPS_Int))
write.csv(inicall_df,"cohort.cnmops.inicall.csv",row.names = FALSE, quote=FALSE)

integerCopyNumber_df<-as.data.frame(integerCopyNumber(resCNMOPS_Int))
write.csv(integerCopyNumber_df,"cohort.cnmops.integerCopyNumber.csv",row.names = FALSE, quote=FALSE)
