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
                    params.rqc_chromosome]).follows(md2).jobs_limit(1,'parallel_task')

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
             pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.rmdup.log")]).follows(index_bam)


        evaluation_intervals = [ x.strip().split("\t") for x in open(params.rqc_evaluation_intervals) if x ]
        coverage_stats_tasks = [] 
        for ev_set in evaluation_intervals : 
            coverage_stats_tasks.append( vc_pipeline.transform(vc_pipeline_utils.coverage_stats, sorted_bam, ruffus.formatter("sort.bam"),
                [pjoin(params.em_vc_output_dir, "{basename[0]}."+ev_set[0]+".coverage.metrics"),
                 pjoin(params.em_vc_output_dir, "logs", "{basename[0]}." + ev_set[0]+".coverage.log")], 
                 extras = [params.em_vc_genome, ev_set[1]], name=f"coverage.{ev_set[0]}").follows(index_bam))

        vc_pipeline.run(multiprocess=params.em_vc_number_of_cpus, logger=logger)

        mark_duplicates_metrics_file = (mark_duplicates_bam._get_output_files(True, []))
        print(mark_duplicates_metrics_file)
        if type(mark_duplicates_metrics_file)==list:
            mark_duplicates_metrics_file = mark_duplicates_metrics_file[0]
        print(mark_duplicates_metrics_file)
        md_metric = vc_pipeline_utils.parse_md_file(mark_duplicates_metrics_file)

        cvg_metrics_files = [ (x._get_output_files(True, []))[0] for x in coverage_stats_tasks ]
        if type(cvg_metrics_files[0]) == list:
            cvg_metrics_files = [ x[0] for x in cvg_metrics_files]

        cvg_metrics = [ vc_pipeline_utils.parse_cvg_metrics( x) for x in er]

        outputs = [vc_pipeline_utils.generate_rqc_output(md_metric, x[0],x[1]) for x in cvg_metrics ]
        summary_df = pd.concat(outputs,axis=1)
        summary_df.columns = [ x[0] for x in ev_set]
        output_hdf_file = pjoin(params.em_vc_output_dir, '.'.join((params.em_vc_basename, "cvg_metrics", "h5")))
        summary_df.to_hdf(output_hdf_file, key="cvg_metrics")
        for i,c in enumerate(cvg_metrics):
            cvg_metrics[1].to_hdf(output_hdf_file, key=f'{ev_set[0]}_histogram')


        print("RapidQC run: success", file=output_log, flush=True)
    except Exception as err : 
        exc_info = sys.exc_info()
        print(*exc_info, file=output_log, flush=True)
        print("RapidQC run: failed", file=output_log, flush=True)
        raise(err)

