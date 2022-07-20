# Copyright 2022 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    CNV calling for a given cohort's reads count.
#    CNV calling is done using cn.mops algorithm documented here : https://academic.oup.com/nar/article/40/9/e69/1136601
# CHANGELOG in reverse chronological order

suppressPackageStartupMessages(library(cn.mops))
suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(rhdf5))

parser <- ArgumentParser()
parser$add_argument("-cohort_rc", "--cohort_reads_count_file",
                    help="input cohort reads count file in Rdata format")
parser$add_argument("-minWidth", "--min_width_val",
                    type="integer",default=2)
parser$add_argument("-p", "--parallel",
                    help="number of parallel processes",
                    type="integer", default=30)
args <- parser$parse_args()

cohort_reads_count_file <- args$cohort_reads_count_file
min_width_val <- args$min_width_val
cores <- args$parallel

#load(cohort_reads_count_file)
readRDS(file = cohort_reads_count_file) #variable name : merged_cohort_reads_count

resCNMOPS <- cn.mops(merged_cohort_reads_count,parallel=cores,minWidth = as.integer(min_width_val) )
resCNMOPS_Int <-calcIntegerCopyNumbers(resCNMOPS)

saveRDS(resCNMOPS_Int,file="cohort.cnmops.rds")

df_cnvs<-as.data.frame(cnvs(resCNMOPS_Int))
write.csv(df_cnvs,"cohort.cnmops.cnvs.csv", row.names = FALSE,quote=FALSE)

# df_cnvr<-as.data.frame(cnvr(resCNMOPS_Int))
# write.csv(df_cnvr,"cohort.cnmops.cnvr.csv", row.names = FALSE,quote=FALSE)
#
# individualCall_df<-as.data.frame(individualCall(resCNMOPS_Int))
# write.csv(individualCall_df,"cohort.cnmops.individualCall.csv",row.names = FALSE, quote=FALSE)
#
# inicall_df<-as.data.frame(iniCall(resCNMOPS_Int))
# write.csv(inicall_df,"cohort.cnmops.inicall.csv",row.names = FALSE, quote=FALSE)
#
# integerCopyNumber_df<-as.data.frame(integerCopyNumber(resCNMOPS_Int))
# write.csv(integerCopyNumber_df,"cohort.cnmops.integerCopyNumber.csv",row.names = FALSE, quote=FALSE)
