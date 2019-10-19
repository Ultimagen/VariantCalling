
import ruffus
import sys
import vc_calling_pipeline_utils
from os.path import join as pjoin

params = vc_calling_pipeline_utils.parse_params_file( sys.argv[1] )
print(params)
vc_pipeline = ruffus.Pipeline(name="Variant calling pipeline")
vc_pipeline.mkdir(params.em_vc_output_dir)
vc_pipeline.mkdir(pjoin(params.em_vc_output_dir, "logs"))
mdtask=vc_pipeline.mkdir(pjoin(params.em_vc_output_dir, "interval_files"))

aln_merge = vc_pipeline.transform(vc_calling_pipeline_utils.align_and_merge, params.em_vc_demux_file, 
	ruffus.formatter(), [pjoin(params.em_vc_output_dir, "{basename[0]}.aln.bam"), 
	pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.aln.log")],
	extras=[params.em_vc_genome, params.em_vc_number_of_cpus]).jobs_limit(1,'parallel_task').follows(mdtask)
fi_file = vc_pipeline.transform(vc_calling_pipeline_utils.prepare_fetch_intervals, params.em_vc_chromosomes_list, 
	ruffus.formatter(),
	pjoin(params.em_vc_output_dir, "filter.bed"), extras=[params.em_vc_genome]).follows(mdtask)

filtered_bam = vc_pipeline.transform(vc_calling_pipeline_utils.fetch_intervals , aln_merge, ruffus.formatter(".aln.bam"), 
	ruffus.add_inputs(fi_file), [pjoin(params.em_vc_output_dir, "{basename[0]}.filter.aln.bam"), 
	                             pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.filter.aln.log")])

sorted_bam = vc_pipeline.transform(vc_calling_pipeline_utils.sort_file, filtered_bam, ruffus.formatter("filter.aln.bam"), 
	[pjoin(params.em_vc_output_dir, "{basename[0]}.sort.bam"), 
     pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.sort.log")], extras=[params.em_vc_number_of_cpus]).jobs_limit(1,'parallel_task')

recalibrated_bam = vc_pipeline.transform(vc_calling_pipeline_utils.recalibrate_file, sorted_bam, ruffus.formatter("sort.bam"), 
	[pjoin(params.em_vc_output_dir, "{basename[0]}.recal.bam"), 
     pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.recal.log")], 
     extras=[params.em_vc_recalibration_model, params.em_vc_number_of_cpus]).jobs_limit(1,'parallel_task')

index_bam = vc_pipeline.transform(vc_calling_pipeline_utils.index_file, recalibrated_bam, ruffus.formatter("recal.bam"), 
	[pjoin(params.em_vc_output_dir, "{basename[0]}.bam.bai"), 
     pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.index.log")])

vc_intervals = vc_pipeline.transform(vc_calling_pipeline_utils.create_variant_calling_intervals, fi_file, ruffus.formatter("filter.bed"), 
	[pjoin(params.em_vc_output_dir, "{basename[0]}.vc.intervals.bed"), 
     pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.vc.intervals.log")])

vc_intervals_split = vc_pipeline.subdivide(vc_calling_pipeline_utils.split_intervals_into_files, vc_intervals, ruffus.formatter(".bed"), 
	pjoin(params.em_vc_output_dir, "interval_files", "{basename[0]}.*"), pjoin(params.em_vc_output_dir, "interval_files", "{basename[0]}")).follows(mdtask)

vc_variant_calls = vc_pipeline.product(vc_calling_pipeline_utils.variant_calling, recalibrated_bam, ruffus.formatter(".bam"), 
										vc_intervals_split, ruffus.formatter(), 
										[pjoin(params.em_vc_output_dir, "{basename[0][0]}{ext[1][0]}.vcf"),
										pjoin(params.em_vc_output_dir, "logs", "{basename[0][0]}{ext[1][0]}.vcf.log"),], 
										extras=[params.em_vc_genome])
vc_pipeline.run(multiprocess=params.em_vc_number_of_cpus)