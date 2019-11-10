
import ruffus
import sys
import vc_pipeline_utils
import python.pipelines.comparison_pipeline as comparison_pipeline
from os.path import join as pjoin
from os.path import splitext

params = vc_pipeline_utils.parse_params_file( sys.argv[1] )
print(params)
vc_pipeline = ruffus.Pipeline(name="Variant calling pipeline")
vc_pipeline.mkdir(params.em_vc_output_dir)
mdtask=vc_pipeline.mkdir(pjoin(params.em_vc_output_dir, "logs"))
mdtask1=vc_pipeline.mkdir(pjoin(params.em_vc_output_dir, "interval_files")).follows(mdtask)

aln_merge = vc_pipeline.transform(vc_pipeline_utils.align_and_merge, params.em_vc_demux_file, 
	ruffus.formatter(), [pjoin(params.em_vc_output_dir, "{basename[0]}.aln.bam"), 
	pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.aln.log")],
	extras=[params.em_vc_genome, params.em_vc_number_of_cpus]).jobs_limit(1,'parallel_task').follows(mdtask1)
fi_file = vc_pipeline.transform(vc_pipeline_utils.prepare_fetch_intervals, params.em_vc_chromosomes_list, 
	ruffus.formatter(),
	pjoin(params.em_vc_output_dir, "filter.bed"), extras=[params.em_vc_genome]).follows(mdtask)

filtered_bam = vc_pipeline.transform(vc_pipeline_utils.fetch_intervals , aln_merge, ruffus.formatter(".aln.bam"), 
	ruffus.add_inputs(fi_file), [pjoin(params.em_vc_output_dir, "{basename[0]}.filter.aln.bam"), 
	                             pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.filter.aln.log")])

sorted_bam = vc_pipeline.transform(vc_pipeline_utils.sort_file, filtered_bam, ruffus.formatter("filter.aln.bam"), 
	[pjoin(params.em_vc_output_dir, "{basename[0]}.sort.bam"), 
     pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.sort.log")], extras=[params.em_vc_number_of_cpus]).jobs_limit(1,'parallel_task')

recalibrated_bam = vc_pipeline.transform(vc_pipeline_utils.recalibrate_file, sorted_bam, ruffus.formatter("sort.bam"), 
	[pjoin(params.em_vc_output_dir, "{basename[0]}.recal.bam"), 
     pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.recal.log")], 
     extras=[params.em_vc_recalibration_model, params.em_vc_number_of_cpus]).jobs_limit(1,'parallel_task')

index_bam = vc_pipeline.transform(vc_pipeline_utils.index_file, recalibrated_bam, ruffus.formatter("recal.bam"), 
	[pjoin(params.em_vc_output_dir, "{basename[0]}.bam.bai"), 
     pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.index.log")])

vc_intervals = vc_pipeline.transform(vc_pipeline_utils.create_variant_calling_intervals, fi_file, ruffus.formatter("filter.bed"), 
	[pjoin(params.em_vc_output_dir, "{basename[0]}.vc.intervals.bed"), 
     pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.vc.intervals.log")]. extras=[params.em_vc_number_of_cpus])

vc_intervals_split = vc_pipeline.subdivide(vc_pipeline_utils.split_intervals_into_files, vc_intervals, ruffus.formatter(".bed"), 
	pjoin(params.em_vc_output_dir, "interval_files", "{basename[0]}.*"), pjoin(params.em_vc_output_dir, "interval_files", "{basename[0]}")).follows(mdtask)

vc_variant_calls = vc_pipeline.product(vc_pipeline_utils.variant_calling, recalibrated_bam, ruffus.formatter(".bam"), 
										vc_intervals_split, ruffus.formatter(), 
										[pjoin(params.em_vc_output_dir, "{basename[0][0]}{ext[1][0]}.vcf"),
										 pjoin(params.em_vc_output_dir, "logs", "{basename[0][0]}{ext[1][0]}.vcf.log")], 
										extras=[params.em_vc_genome]).follows(index_bam)

vc_pipeline.run(multiprocess=params.em_vc_number_of_cpus)


comparison_interval_files = vc_pipeline_utils.generate_comparison_intervals( 
														pjoin(params.em_vc_output_dir, "filter.bed"), 
														params.em_vc_genome, params.em_vc_output_dir)

recalibrated_bam_name = recalibrated_bam._get_output_files(True, [])[0]
if type(recalibrated_bam_name) == list : 
	recalibrated_bam_name = recalibrated_bam_name[0]

header_file = vc_pipeline_utils.generate_header(
	'.'.join((splitext(recalibrated_bam_name)[0], '1','vcf')),
	'h1.hdr', params.em_vc_output_dir)

for ci_file in comparison_interval_files : 

	concordance, results = comparison_pipeline.pipeline(len(comparison_interval_files)*params.em_vc_number_of_cpus, 
	    splitext(recalibrated_bam_name)[0],
	    header_file, 
	    truth_file=params.em_vc_ground_truth, 
	    highconf_intervals=params.em_vc_ground_truth_highconf,
	    runs_intervals=params.em_vc_gaps_hmers_filter, 
	    cmp_intervals=ci_file, 
	    ref_genome=params.em_vc_genome,
	    call_sample='sm1', 
	    truth_sample='sm1')
	concordance.to_hdf('.'.join((splitext(ci_file)[0], 'h5')), key='concordance', mode='w')
	results.to_hdf('.'.join((splitext(ci_file)[0], 'h5')), key='results', mode='a')

