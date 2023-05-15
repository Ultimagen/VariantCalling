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
#    Combine reads counts of several input samples into a single GenomicRanges object.
# CHANGELOG in reverse chronological order

suppressPackageStartupMessages(library(cn.mops))
suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(rhdf5))

parser <- ArgumentParser()
parser$add_argument("-samples_read_count_files_list", "--samples_read_count_files_list",
                    help="file containing a list of all reads count rds files per sample")
parser$add_argument("--save_csv", action='store_true',
                    help="whether to save reads count data-frames in csv format")
parser$add_argument("--save_hdf", action='store_true',
                    help="whether to save reads count data-frames in hdf5 format")
args <- parser$parse_args()

# ------------------------------------------------------------------------- #
#' Creating merged Reads Counts object from all samples in cohort
# ------------------------------------------------------------------------- #
RC_RData_files = scan(args$samples_read_count_files_list, what="", sep="\n")

gr1<-readRDS(file = RC_RData_files[1])
gr2<-readRDS(file = RC_RData_files[2])

gr <- GRanges(
  seqnames = seqnames(gr1),
  ranges = ranges(gr1),
  strand = strand(gr1),
  mcols(gr1),
  mcols(gr2)
)

for(i in 3:length(RC_RData_files)){
  gr_new<-readRDS(file=RC_RData_files[i])

  gr <- GRanges(
    seqnames = seqnames(gr),
    ranges = ranges(gr),
    strand = strand(gr),
    mcols(gr),
    mcols(gr_new)
  )
}

saveRDS(gr,file="merged_cohort_reads_count.rds")
if(args$save_csv){
  write.csv(as.data.frame(gr),paste(path_to_RC.RData_files,"merged_cohort_reads_count.csv",sep = ""), row.names = FALSE)
}
if(args$save_hdf){
  h5write(as.data.frame(gr),"merged_cohort_reads_count.hdf5","merged_cohort_reads_count")
}

