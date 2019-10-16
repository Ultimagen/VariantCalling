
import ruffus
import sys
import vc_calling_pipeline_utils
from os.path import join as pjoin

params = vc_calling_pipeline_utils.parse_params_file( sys.argv[1] )
print(params)
vc_pipeline = ruffus.Pipeline(name="Variant calling pipeline")
vc_pipeline.mkdir(params.output_dir)
vc_pipeline.mkdir(pjoin(params.output_dir, "logs"))
mdtask=vc_pipeline.mkdir(pjoin(params.output_dir, "interval_files"))

aln_merge = vc_pipeline.transform(vc_calling_pipeline_utils.align_and_merge, params.demux_file, 
	ruffus.formatter(), [pjoin(params.output_dir, "{basename[0]}.aln.bam"), 
	pjoin(params.output_dir, "logs", "{basename[0]}.aln.log")],
	extras=[params.genome])
fi_file = vc_pipeline.transform(vc_calling_pipeline_utils.prepare_fetch_intervals, params.chromosomes_list, 
	ruffus.formatter(),
	pjoin(params.output_dir, "filter.bed"), extras=[params.genome])

filtered_bam = vc_pipeline.transform(vc_calling_pipeline_utils.fetch_intervals , aln_merge, ruffus.formatter(".aln.bam"), 
	ruffus.add_inputs(fi_file), [pjoin(params.output_dir, "{basename[0]}.filter.aln.bam"), 
	                             pjoin(params.output_dir, "logs", "{basename[0]}.filter.aln.log")])

sorted_bam = vc_pipeline.transform(vc_calling_pipeline_utils.sort_file, filtered_bam, ruffus.formatter("filter.aln.bam"), 
	[pjoin(params.output_dir, "{basename[0]}.sort.bam"), 
     pjoin(params.output_dir, "logs", "{basename[0]}.sort.log")])

recalibrated_bam = vc_pipeline.transform(vc_calling_pipeline_utils.recalibrate_file, sorted_bam, ruffus.formatter("sort.bam"), 
	[pjoin(params.output_dir, "{basename[0]}.recal.bam"), 
     pjoin(params.output_dir, "logs", "{basename[0]}.recal.log")], extras=[params.recalibration_model])

index_bam = vc_pipeline.transform(vc_calling_pipeline_utils.index_file, recalibrated_bam, ruffus.formatter("recal.bam"), 
	[pjoin(params.output_dir, "{basename[0]}.bam.bai"), 
     pjoin(params.output_dir, "logs", "{basename[0]}.index.log")])

vc_intervals = vc_pipeline.transform(vc_calling_pipeline_utils.create_variant_calling_intervals, fi_file, ruffus.formatter("filter.bed"), 
	[pjoin(params.output_dir, "{basename[0]}.vc.intervals.bed"), 
     pjoin(params.output_dir, "logs", "{basename[0]}.vc.intervals.log")])

vc_intervals_split = vc_pipeline.subdivide(vc_calling_pipeline_utils.split_intervals_into_files, vc_intervals, ruffus.formatter(".bed"), 
	pjoin(params.output_dir, "interval_files", "{basename[0]}.*"), pjoin(params.output_dir, "interval_files", "{basename[0]}")).follows(mdtask)

vc_variant_calls = vc_pipeline.product(vc_calling_pipeline_utils.variant_calling, recalibrated_bam, ruffus.formatter(".bam"), 
										vc_intervals_split, ruffus.formatter(), 
										[pjoin(params.output_dir, "{basename[0][0]}{ext[1][0]}"),
										pjoin(params.output_dir, "logs", "{basename[0][0]}{ext[1][0]}.log"),], 
										extras=[params.genome])
vc_pipeline.run()