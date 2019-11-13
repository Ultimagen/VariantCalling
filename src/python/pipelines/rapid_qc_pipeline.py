import ruffus
import ruffus.task
import sys
import vc_pipeline_utils

from os.path import join as pjoin
from os.path import basename, dirname
from os import mkdir
import pandas as pd
import traceback

params = vc_pipeline_utils.parse_params_file( sys.argv[1], "rapidqc" )
logname = '.'.join((params.em_vc_basename, "rqc.log"))

try:
    mkdir(params.em_vc_output_dir)
except FileExistsError: 
    pass


with open(pjoin(params.em_vc_output_dir, logname),'w') as output_log : 
    logger = ruffus.task.t_stream_logger(output_log)
    print("Running with the following parameters", file=output_log, flush=False )
    print("-------------------------------------\n\n", file=output_log, flush=True)
    for key in params._get_kwargs() : 
        print(f"{key[0]} -> {key[1]}", file=output_log, flush=True)
    try: 
        vc_pipeline = ruffus.Pipeline(name="Error rate estimation pipeline")
        md1 = vc_pipeline.mkdir(params.em_vc_output_dir)
        md2 = vc_pipeline.mkdir(pjoin(params.em_vc_output_dir, "logs"))
        aln = vc_pipeline.transform(vc_pipeline_utils.align_minimap_and_filter, 
                    params.em_vc_demux_file, ruffus.formatter(), 
                    [pjoin(params.em_vc_output_dir, "{basename[0]}.aln.bam"), 
                    pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.aln.log")],
                    extras=[params.em_vc_genome, params.em_vc_number_of_cpus, 
                    params.em_vc_chromosomes_list]).follows(md2).jobs_limit(1,'parallel_task')

        sorted_bam = vc_pipeline.transform(vc_pipeline_utils.sort_file, aln, ruffus.formatter("aln.bam"), 
            [pjoin(params.em_vc_output_dir, "{basename[0]}.sort.bam"), 
             pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.sort.log")], 
             extras=[ params.em_vc_number_of_cpus]).jobs_limit(1,'parallel_task')

        index_bam = vc_pipeline.transform(vc_pipeline_utils.index_file, sorted_bam, ruffus.formatter("sort.bam"), 
            [pjoin(params.em_vc_output_dir, "{basename[0]}.bam.bai"), 
             pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.index.log")])

        mark_duplicates_bam = vc_pipeline.transform(vc_pipeline_utils.mark_duplicates, sorted_bam, ruffus.formatter("sort.bam"), 
            [pjoin(params.em_vc_output_dir, "{basename[0]}.rmdup.bam"),
             pjoin(params.em_vc_output_dir, "{basename[0]}.rmdup.metrics"),
             pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.index.log")]).follows(index_bam)


        coverage_stats = vc_pipeline.transform(vc_pipeline_utils.coverage_stats, sorted_bam, ruffus.formatter("sort.bam"), 
            [pjoin(params.em_vc_output_dir, "{basename[0]}.coverage.metrics"),
             pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.coverage.log")], 
             extras = [params.em_vc_genome, params.rqc_evaluation_intervals]).follows(index_bam)
        vc_pipeline.run(multiprocess=params.em_vc_number_of_cpus, logger=logger)

    except Exception as err : 
        exc_info = sys.exc_info()
        print(*exc_info, file=output_log, flush=True)
        print("FastQC run: failed", file=output_log, flush=True)
        raise(err)

