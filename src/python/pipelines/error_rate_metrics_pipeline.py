import ruffus
import sys
import vc_calling_pipeline_utils
from os.path import join as pjoin

params = vc_calling_pipeline_utils.parse_params_file( sys.argv[1] )
print(params)
vc_pipeline = ruffus.Pipeline(name="Error rate estimation pipeline")
vc_pipeline.mkdir(params.output_dir)
vc_pipeline.mkdir(pjoin(params.output_dir, "logs"))

head_file = vc_pipeline.transform(vc_calling_pipeline_utils.head_file, params.demux_file, ruffus.formatter(), 
	[pjoin(params.output_dir, "{basename[0]}.head.bam"),
	pjoin(params.output_dir, "logs", "{basename[0]}.head.log")], params.number_to_sample)

aln = vc_pipeline.transform(vc_calling_pipeline_utils.align, head_file, 
	ruffus.formatter(), [pjoin(params.output_dir, "{basename[0]}.aln.bam"), 
	pjoin(params.output_dir, "logs", "{basename[0]}.aln.log")],
	extras=[params.genome])

filtered_bam = vc_pipeline.transform(vc_calling_pipeline_utils.filter_quality , aln, ruffus.formatter(".aln.bam"), 
	[pjoin(params.output_dir, "{basename[0]}.filter.aln.bam"), 
	                             pjoin(params.output_dir, "logs", "{basename[0]}.filter.aln.log")])

sorted_bam = vc_pipeline.transform(vc_calling_pipeline_utils.sort_file, filtered_bam, ruffus.formatter("filter.aln.bam"), 
	[pjoin(params.output_dir, "{basename[0]}.sort.bam"), 
     pjoin(params.output_dir, "logs", "{basename[0]}.sort.log")])

index_bam = vc_pipeline.transform(vc_calling_pipeline_utils.index_file, sorted_bam, ruffus.formatter("sort.bam"), 
	[pjoin(params.output_dir, "{basename[0]}.bam.bai"), 
     pjoin(params.output_dir, "logs", "{basename[0]}.index.log")])

error_metrics = vc_pipeline.transform(vc_calling_pipeline_utils.error_metrics, sorted_bam, 
	ruffus.formatter("sort.bam"), [pjoin(params.output_dir, "{basename[0]}.metrics"), 
     							   pjoin(params.output_dir, "logs", "{basename[0]}.metric.log")],
     							   extras=[params.genome])
vc_pipeline.run()