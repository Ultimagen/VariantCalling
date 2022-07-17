suppressPackageStartupMessages(library(cn.mops))
suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(argparse))

parser <- ArgumentParser()
parser$add_argument("-cohort_rc", "--cohort_reads_count_file",
                    help="input cohort reads count file in Rdata format")
parser$add_argument("-minWidth", "--min_width_val",
                    help="input sample reads count file in Rdata format",
                    type="integer",default=2)
parser$add_argument("-p", "--parallel",
                    help="number of parallel process",
                    type="integer", default=30)
args <- parser$parse_args()

cohort_reads_count_file <- args$cohort_reads_count_file
min_width_val <- args$min_width_val
cores <- args$parallel

load(cohort_reads_count_file) #variable name : merged_cohort_reads_count

resCNMOPS <- cn.mops(merged_cohort_reads_count,parallel=cores,minWidth = as.integer(min_width_val) )
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
