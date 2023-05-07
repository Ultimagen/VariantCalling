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
#    Add reads count of a single sample to a cohort's reads count object.
# CHANGELOG in reverse chronological order

suppressPackageStartupMessages(library(cn.mops))
suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(rhdf5))

parser <- ArgumentParser()
parser$add_argument("-cohort_rc", "--cohort_reads_count_file",
                    help="input cohort reads count file in rds format")
parser$add_argument("-sample_rc", "--sample_reads_count_file",
                    help="input sample reads count file in rds format")
parser$add_argument("--save_hdf", action='store_true',
                    help="whether to save reads count data-frames in hdf5 format")
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
if(args$save_hdf){
  h5write(as.data.frame(merged_cohort_reads_count),"merged_cohort_reads_count.hdf5","merged_cohort_reads_count")
}
