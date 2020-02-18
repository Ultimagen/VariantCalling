import ruffus
import ruffus.task
import sys
from os.path import join as pjoin
from os import mkdir
import pandas as pd
import pathmagic # noqa
import vc_pipeline_utils

params = vc_pipeline_utils.parse_params_file("error_metrics")
logname = '.'.join((params.em_vc_basename, "em.log"))


try:
    mkdir(params.em_vc_output_dir)
except FileExistsError:
    pass


with open(pjoin(params.em_vc_output_dir, logname), 'w') as output_log:
    logger = ruffus.task.t_stream_logger(output_log)
    print("Running with the following parameters", file=output_log, flush=False)
    print("-------------------------------------\n\n", file=output_log, flush=True)
    for key in params._get_kwargs():
        print(f"{key[0]} -> {key[1]}", file=output_log, flush=True)
    try:
        vc_pipeline = ruffus.Pipeline(name="Error rate estimation pipeline")
        md1 = vc_pipeline.mkdir(params.em_vc_output_dir)
        md2 = vc_pipeline.mkdir(pjoin(params.em_vc_output_dir, "logs"))
        if params.em_vc_number_to_sample >= 0:
            head_file = vc_pipeline.transform(vc_pipeline_utils.head_file, params.em_vc_demux_file, ruffus.formatter(),
                                              [pjoin(params.em_vc_output_dir, "{basename[0]}.head.bam"),
                                               pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.head.log")],
                                              extras=[params.em_vc_number_to_sample,
                                                      params.em_vc_number_of_cpus]).\
                follows(md2).jobs_limit(1, 'parallel_task')

            aln = vc_pipeline.transform(vc_pipeline_utils.align, head_file,
                                        ruffus.formatter(),
                                        [pjoin(params.em_vc_output_dir, "{basename[0]}.aln.bam"),
                                         pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.aln.log")],
                                        extras=[params.em_vc_genome,
                                                params.em_vc_number_of_cpus]).jobs_limit(1, 'parallel_task')

        else:

            aln = vc_pipeline.transform(vc_pipeline_utils.align, params.em_vc_demux_file,
                                        ruffus.formatter(),
                                        [pjoin(params.em_vc_output_dir, "{basename[0]}.aln.bam"),
                                         pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.aln.log")],
                                        extras=[params.em_vc_genome,
                                                params.em_vc_number_of_cpus])\
                .follows(md2).jobs_limit(1, 'parallel_task')

        sorted_bam = vc_pipeline.transform(vc_pipeline_utils.sort_file, aln, ruffus.formatter("aln.bam"),
                                           [pjoin(params.em_vc_output_dir, "{basename[0]}.sort.bam"),
                                            pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.sort.log")],
                                           extras=[params.em_vc_number_of_cpus]).jobs_limit(1, 'parallel_task')

        index_bam = vc_pipeline.transform(vc_pipeline_utils.index_file, sorted_bam, ruffus.formatter("sort.bam"),
                                          [pjoin(params.em_vc_output_dir, "{basename[0]}.bam.bai"),
                                           pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.index.log")])

        filtered_bam = vc_pipeline.transform(vc_pipeline_utils.filter_quality,
                                             sorted_bam, ruffus.formatter("aln.sort.bam"),
                                             [pjoin(params.em_vc_output_dir, "{basename[0]}.filter.bam"),
                                              pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.filter.log")],
                                             extras=[params.em_vc_number_of_cpus]).jobs_limit(1, 'parallel_task')

        index_bam_1 = vc_pipeline.transform(vc_pipeline_utils.index_file, filtered_bam, ruffus.formatter("filter.bam"),
                                            [pjoin(params.em_vc_output_dir, "{basename[0]}.bam.bai"),
                                             pjoin(params.em_vc_output_dir,
                                                   "logs", "{basename[0]}.index1.log")], name="idx1")

        error_metrics_q20 = vc_pipeline.transform(vc_pipeline_utils.error_metrics, filtered_bam,
                                                  ruffus.formatter("filter.bam"),
                                                  [pjoin(params.em_vc_output_dir, "{basename[0]}.metrics"),
                                                   pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.metric.log")],
                                                  extras=[params.em_vc_genome]).follows(index_bam_1)

        error_metrics_q0 = vc_pipeline.transform(vc_pipeline_utils.error_metrics, sorted_bam,
                                                 ruffus.formatter("sort.bam"),
                                                 [pjoin(params.em_vc_output_dir, "{basename[0]}.metrics"),
                                                  pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.metric.log")],
                                                 extras=[params.em_vc_genome], name="metrics_q0").follows(index_bam)

        idxstats_metrics = vc_pipeline.transform(vc_pipeline_utils.idxstats, sorted_bam,
                                                 ruffus.formatter("sort.bam"),
                                                 [pjoin(params.em_vc_output_dir, "{basename[0]}.idxstats"),
                                                  pjoin(params.em_vc_output_dir,
                                                        "logs", "{basename[0]}.idxstats.log")]).follows(index_bam)

        ftrt: list = []
        if params.em_vc_rerun_all and params.em_vc_number_to_sample >= 0:
            ftrt += [md1, md2, head_file]
        elif params.em_vc_rerun_all:
            ftrt += [md1, md2, aln]

        vc_pipeline.run(multiprocess=params.em_vc_number_of_cpus, logger=logger, forcedtorun_tasks=ftrt)

        # Parsing and processing
        idxstats_metrics_file = vc_pipeline_utils.flatten(idxstats_metrics._get_output_files(True, []))
        idxstats_metrics_file = [x for x in idxstats_metrics_file if x.endswith("idxstats")][0]

        error_metrics_q0_file = vc_pipeline_utils.flatten(error_metrics_q0._get_output_files(True, []))
        error_metrics_q0_file = [x for x in error_metrics_q0_file if x.endswith("metrics")][0]

        error_metrics_q20_file = vc_pipeline_utils.flatten(error_metrics_q20._get_output_files(True, []))
        error_metrics_q20_file = [x for x in error_metrics_q20_file if x.endswith("metrics")][0]

        idxstats_df = vc_pipeline_utils.collect_alnstats(idxstats_metrics_file, error_metrics_q20_file)
        q0_df, complete_df = vc_pipeline_utils.collect_metrics(error_metrics_q0_file)
        q20_df, complete_df = vc_pipeline_utils.collect_metrics(error_metrics_q20_file)
        em_df = pd.concat((q0_df, q20_df), axis=1)
        em_df.columns = (['Unfiltered', 'Q20 filtered'])
        output_hdf_file = pjoin(params.em_vc_output_dir, '.'.join((params.em_vc_basename, "bwa_metrics", "h5")))
        idxstats_df.to_hdf(output_hdf_file, key="bwa_alignment_stats")
        em_df.to_hdf(output_hdf_file, key="bwa_error_rates")
        complete_df.to_hdf(output_hdf_file, key="bwa_all_metrics")
        print("Error metrics run: success", file=output_log, flush=True)

    except Exception as err:
        exc_info = sys.exc_info()
        print(*exc_info, file=output_log, flush=True)
        print("Error metrics run: failed", file=output_log, flush=True)
        raise(err)
