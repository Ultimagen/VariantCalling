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
#    whole genome reads count normalization for a given cohort.
#    reads count normalization function is based on cn.mops package documented here : https://academic.oup.com/nar/article/40/9/e69/1136601
# CHANGELOG in reverse chronological order

suppressPackageStartupMessages(library(cn.mops))
suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(argparse))


.statmod <- function(x,na.rm=FALSE) {
	if (na.rm){
		z <- table(as.vector(x[!is.na(x)]))
		r <- names(z)[z == max(z)]
		return(as.numeric(r)[1])
	} else {
		if (any(is.na(x))){return(NA)
		} else {
			z <- table(as.vector(x))
			r <- names(z)[z == max(z)]
			return(as.numeric(r)[1])
		}
	}
}

GetVarianceSum <- function(X,in_cutoff){
  left_dist = X[X<=in_cutoff]
  right_dist = X[X>in_cutoff]
  left_variance = var(left_dist)
  right_variance = var(right_dist)
  sum_var = left_variance+right_variance
  return(sum_var)
}

#' @title whole Genome-wize Normalization of NGS data.
#'
#' @description Normalize quantitative NGS data in order to make counts comparable over
#' samples, i.e., correcting for different library sizes or coverages.
#' Scales each samples' reads such that the coverage is even for
#' all samples after normalization.
#'
#' @param X Matrix of positive real values, where
#' columns are interpreted as samples and rows as genomic regions. An entry is
#' the read count of a sample in the genomic region. Alternatively this can be
#' a GRanges object containing the read counts as values.
#' @param chr Character vector that has as many elements as "X" has rows. The
#' vector assigns each genomic segment to a reference sequence (chromosome).
#' @param normType Type of the normalization technique. Each samples'
#' read counts are scaled such that the total number of reads are comparable across
#' samples.
#' If this parameter is set to the value "mode",
#' the read counts are scaled such that each samples'
#' most frequent value (the "mode") is equal after normalization.
#' Accordingly for the other options are "mean","median","poisson", "quant", and "mode".
#' Default = "poisson".
#' @param sizeFactor  By this parameter one can decide to how the size factors
#' are calculated.
#' Possible choices are the the mean, median or mode coverage ("mean", "median", "mode") or any quantile
#' ("quant").
#' @param qu Quantile of the normType if normType is set to "quant" .Real value between 0 and 1. Default = 0.25.
#' @param quSizeFactor Quantile of the sizeFactor if sizeFactor is set to "quant".
#' 0.75 corresponds to "upper quartile normalization". Real value between 0 and 1. Default = 0.75.
#' @param ploidy An integer value for each sample or each column in the read
#' count matrix. At least two samples must have a ploidy of 2. Default = "missing".
#' @examples
#' data(cn.mops)
#' X.norm <- normalizeChromosomes(X)
#' @return A data matrix of normalized read counts with the same dimensions
#' as the input matrix X.
#' @author Tammy Biniashvili \email{tammy.biniashvili@ultimagen.com}.
#' original script source is cn.mops repository: https://github.com/bioinf-jku/cn.mops/blob/f9fef3f2c6a5a0bd323a754a0256b840565401ba/R/normalizeChromosomes.R
#' @export


normalizeChromosomesGenomewize <- function(X, chr, normType="poisson", sizeFactor="mean",
		qu=0.25, quSizeFactor=0.75,chr_X_name = "chrX", chr_Y_name="chrY", ploidy){

	if (!(normType %in% c("mean","median","poisson","mode", "quant"))){
		stop(paste("Set TO of normalization to \"mean\"",
						"\"median\", \"quant\" or \"mode\"."))
	}
	if (!(sizeFactor %in% c("mean","median","quant","mode"))){
		stop(paste("Set TO of normalization to \"mean\"",
						"\"median\", \"quant\" or \"mode\"."))
	}
	input <- X
	if (!(is.matrix(input)|any(class(input)=="GRanges"))){
		stop("Input data must be matrix or GRanges object!")
	}
	returnGRanges <- FALSE
	if(any(class(X)=="GRanges")){
		returnGRanges <- TRUE
		X <- as.matrix(mcols(X))
	}
	if (is.vector(X)){X <- matrix(X,nrow=1)}

	if (missing(chr) & is.matrix(input)){
		chr <- rep("undef",nrow(X))
	}
	if (missing(chr) & any(class(input)=="GRanges")){
		chr <- as.character(seqnames(input))
	}

	Xorig <- X

	# Sequencing data matrix
	# vector of chromosome - length equal to rows of X
	if (length(chr)!=nrow(X)){
		stop("Length of \"chr\" must be equal to number of rows of \"X\".")}
	chr <- (as.character(chr))

	YY <- matrix(0,nrow=nrow(Xorig),ncol=ncol(Xorig))

	  X <- Xorig
		globalsizeFactors <- colSums(X,na.rm=TRUE)

		if (any(is.na(globalsizeFactors))) stop("NAs in readcount matrix.")
		if (any(globalsizeFactors==0)) stop("Zero columns in readcount matrix.")

		if (ncol(X)==1){
			Y <- X
		} else {

			Y <- matrix(0,nrow=nrow(X),ncol=ncol(X))
			chrIdx <- which((chr != chr_X_name) & (chr != chr_Y_name))
			Ytmp <- X[chrIdx, ,drop=FALSE]

			if (all(Ytmp==0)){
				Y[chrIdx, ] <- Ytmp
			} else {
				idxSG <- apply(Ytmp,1,function(x) all(x<1))
				Ytmp[idxSG, ] <- NA

				if ((nrow(Ytmp)-length(which(idxSG))) > 1){

					if (sizeFactor=="mean"){
						sizeFactors <- colMeans(Ytmp,na.rm=TRUE)
					} else if (sizeFactor=="median" | normType=="poisson"){
						sizeFactors <- apply(Ytmp,2,median,na.rm=TRUE)
					} else if (sizeFactor=="quant"){
						sizeFactors <- apply(Ytmp,2,quantile,probs=quSizeFactor,na.rm=TRUE)
					} else if (sizeFactor=="mode"){
						sizeFactors <- apply(Ytmp,2,function(x) .statmod(x[x!=0],na.rm=TRUE) )
					}

					if (any(is.na(sizeFactors))){
						warning(paste("Normalization failed for reference sequence ",l,
										". Using global sizeFactors!"))
						sizeFactors <- globalsizeFactors
					}

					if (any(sizeFactors==0)){
						stop(paste("Some normalization factors are zero!",
										"Remove samples or chromosomes for which the average read count is zero,",
										"e.g. chromosome Y."))
					}


						if (normType=="mean"){
							correctwiththis <-  mean(sizeFactors,na.rm=TRUE)/sizeFactors
						} else if (normType=="median"){
							correctwiththis <-  median(sizeFactors,na.rm=TRUE)/sizeFactors
						} else if (normType=="quant"){
							correctwiththis <-  quantile(sizeFactors,probs=qu,na.rm=TRUE)/sizeFactors
						} else if (normType=="mode"){
							asm <- .statmod(Ytmp[Ytmp!=0],na.rm=TRUE)
							correctwiththis <- asm/sizeFactors
						} else if (normType=="poisson"){
							correctwiththis <-  mean(sizeFactors,na.rm=TRUE)/sizeFactors
							#YYtmp <- t(t(Ytmp)*correctwiththis)
							YYtmp <- Ytmp %*% diag(correctwiththis)
							v2m <- apply(YYtmp,1,var)/rowMeans(YYtmp)
							uut <- quantile(v2m,probs=0.95,na.rm=TRUE)
							dd <- density(v2m,na.rm=TRUE,from=0,
									to=uut,n=uut*100)
							#mv2m <- median(v2m,na.rm=TRUE)
							mv2m <- dd$x[which.max(dd$y)]
							if (is.finite(mv2m)) {correctwiththis <- correctwiththis*1/mv2m}
						}
						#browser()

					} else {
						warning(paste("Normalization for reference sequence ",l,"not",
										"applicable, because of low number of segments"))
						correctwiththis <- rep(1,ncol(X))
					}

					if (any(!is.finite(correctwiththis))){
						warning(paste("Normalization for reference sequence ",l,"not",
										"applicable, because at least one sample has zero",
										"reads."))
						correctwiththis <- rep(1,ncol(X))
					}

					Ytmp <- X %*% diag(correctwiththis)
					Ytmp[idxSG, ] <- 0

					Y <- Ytmp
				}

		} # over chr

	#handle Sex chromosomes(chrX,chrY) by Gender ploidy value.
	#for male we will multiply normalized X and Y coverage with 2.
	if(length(chr[chr == chr_X_name])>1)
	{
		if (missing(ploidy)){
			ploidy <- rep(2,ncol(X))
			chrIdx <- which(chr == chr_X_name)
			chrX_matrix <-Y[chrIdx,]
			chrX_matrix_nonZero <- chrX_matrix[which(rowSums(chrX_matrix) > 0),]
			chrX_means <- colMeans(chrX_matrix_nonZero,na.rm=TRUE)

			cutoff=min(chrX_means)+1
			sum_var=GetVarianceSum(chrX_means,cutoff)
			cutoff_range = seq(from = min(chrX_means)+1, to = max(chrX_means), by = 1)

			for (cutoff_tmp in cutoff_range) {
				sum_var_tmp = GetVarianceSum(chrX_means,cutoff_tmp)
				if(sum_var_tmp < sum_var) {
				  sum_var = sum_var_tmp
				  cutoff = cutoff_tmp
				}
			}
			ploidy[chrX_means<=cutoff]<-1
			write.table(ploidy,paste(out_prefix,".estimate_gender",sep=""),row.names = FALSE,quote=FALSE,col.names = FALSE)

			png(paste(out_prefix,"chrX_mean_coverage_distribution.png"))
			hist(chrX_means)
			abline(v=cutoff, col='red', lwd=3, lty='dashed')
			dev.off()
		}


		if (any(ploidy!=as.integer(ploidy))){
			stop("Ploidy values must be integers!")
		}
		if (length(ploidy)!=ncol(X)){
			stop("Length of the ploidy vector does not match the number of",
					"columns of the read count matrix!")
		}
		ploidy <- as.integer(ploidy)
		if (length(unique(ploidy))==1) ploidy <- rep(2, ncol(X))
		if (!length(which(ploidy>=2))){
			stop("At least two diploid samples must be contained in the data.")
		}
		#ploidy[ploidy==0] <- 0.05

		ploidy2median <- median(Y[!idxSG,ploidy==2],na.rm=TRUE)
		for (pp in unique(ploidy)){
			if (pp!=2){
				mm <- median(Y[!idxSG, ],na.rm=TRUE)
				if (ploidy2median==0 & mm==0){
					YY[,ploidy==pp] <- Y[,ploidy==pp]
				} else {
				  chrIdx <- which((chr == chr_X_name) | (chr == chr_Y_name))
					YY[chrIdx,ploidy==pp] <- Y[chrIdx,ploidy==pp]*2
					chrIdx <- which((chr != chr_X_name) & (chr != chr_Y_name))
					YY[chrIdx,ploidy==pp] <- Y[chrIdx,ploidy==pp]
				}
			} else{
				YY[,ploidy==pp] <- Y[,ploidy==pp]
			}
		}
	}
	else {
		YY<-Y
	}

	rownames(YY) <- rownames(Xorig)
	colnames(YY) <- colnames(Xorig)

	if (returnGRanges){
		mcols(input) <- YY
		return(input)
	} else {
		return(YY)
	}
}


parser <- ArgumentParser()
parser$add_argument("-cohort_reads_count_file", "--cohort_reads_count_file",
                    help="input cohort reads count file in rds format")
parser$add_argument("-out_prefix", "--out_prefix",
					help="out prefix",
					default="")
parser$add_argument("-ploidy", "--ploidy",
					help="ploidy file decoding cohort's sample gender as 1-male 2-female",
					required = FALSE)
parser$add_argument("-chrX_name", "--chrX_name",default = "chrX",
					help="chrX_name, default: chrX",
					required = FALSE)
parser$add_argument("-chrY_name", "--chrY_name", default = "chrY",
					help="chrY_name, default: chrY",
					required = FALSE)
parser$add_argument("--save_csv", action='store_true',
                    help="whether to save normalized reads count data-frames in csv format")


args <- parser$parse_args()

cohort_reads_count_file <- args$cohort_reads_count_file
out_prefix <- args$out_prefix
ploidy_file <-args$ploidy
chrX_name <- args$chrX_name
chrY_name <- args$chrY_name

cohort_reads_count<- readRDS(file = cohort_reads_count_file )
if (!is.null(ploidy_file)) {
	ploidy_vector<-readLines(ploidy_file)
	cohort_reads_count_normalized <- normalizeChromosomesGenomewize(cohort_reads_count,ploidy=ploidy_vector,chr_X_name=chrX_name,chr_Y_name=chrY_name)
	saveRDS(cohort_reads_count_normalized,paste(out_prefix,"cohort_reads_count.norm.rds",sep=""))
	if(args$save_csv){
		write.csv(as.data.frame(cohort_reads_count_normalized),paste(out_prefix,"cohort_reads_count.norm.csv",sep=""), row.names = FALSE,quote=FALSE)
	}
} else {
	print("Sex will be estimated by chrX coverage")
	cohort_reads_count_normalized <- normalizeChromosomesGenomewize(cohort_reads_count,chr_X_name=chrX_name,chr_Y_name=chrY_name)
	saveRDS(cohort_reads_count_normalized,paste(out_prefix,"cohort_reads_count.norm.rds",sep=""))
	if(args$save_csv){
		write.csv(as.data.frame(cohort_reads_count_normalized),paste(out_prefix,"cohort_reads_count.norm.csv",sep=""), row.names = FALSE,quote=FALSE)
	}
}
