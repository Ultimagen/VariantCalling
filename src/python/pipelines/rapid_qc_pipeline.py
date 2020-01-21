import ruffus
import ruffus.task
import sys
import vc_pipeline_utils

from os.path import join as pjoin
from os import mkdir
import pandas as pd
from psutil import virtual_memory

params = vc_pipeline_utils.parse_params_file(sys.argv[1], "rapidqc")
logname = '.'.join((params.em_vc_basename, "rqc.log"))

try:
    mkdir(params.em_vc_output_dir)
except FileExistsError:
    pass


with open(pjoin(params.em_vc_output_dir, logname), 'w', buffering=1) as output_log:
    logger = ruffus.task.t_stream_logger(output_log)
    print("Running with the following parameters", file=output_log, flush=False)
    print("-------------------------------------\n\n",
          file=output_log, flush=True)
    for key in params._get_kwargs():
        print(f"{key[0]} -> {key[1]}", file=output_log, flush=True)
    try:
        vc_pipeline = ruffus.Pipeline(name="Error rate estimation pipeline")
        md1 = vc_pipeline.mkdir(params.em_vc_output_dir)
        md2 = vc_pipeline.mkdir(pjoin(params.em_vc_output_dir, "logs"))
        aln = vc_pipeline.transform(vc_pipeline_utils.align_minimap_and_filter,
                                    params.em_vc_demux_file, ruffus.formatter(),
                                    [pjoin(params.em_vc_output_dir, "{basename[0]}.rqc.aln.bam"),
                                     pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.rqc.aln.log")],
                                    extras=[params.em_vc_genome, params.em_vc_number_of_cpus,
                                            params.rqc_chromosome]).follows(md2).jobs_limit(1, 'parallel_task')
        count_reads = vc_pipeline.transform(vc_pipeline_utils.extract_total_n_reads, aln,
                                            ruffus.formatter("rqc.aln.bam"),
                                            [pjoin(params.em_vc_output_dir, "{basename[0]}.read_count.txt")])

        sorted_bam = vc_pipeline.transform(vc_pipeline_utils.sort_file, aln, ruffus.formatter("aln.bam"),
                                           [pjoin(params.em_vc_output_dir, "{basename[0]}.sort.bam"),
                                            pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.sort.log")],
                                           extras=[params.em_vc_number_of_cpus]).jobs_limit(1, 'parallel_task')

        index_bam = vc_pipeline.transform(vc_pipeline_utils.index_file, sorted_bam, ruffus.formatter("sort.bam"),
                                          [pjoin(params.em_vc_output_dir, "{basename[0]}.bam.bai"),
                                           pjoin(params.em_vc_output_dir, "logs", "{basename[0]}.index.log")])

        mark_duplicates_bam = vc_pipeline.transform(vc_pipeline_utils.mark_duplicates,
                                                    sorted_bam, ruffus.formatter(
                                                        "sort.bam"),
                                                    [pjoin(params.em_vc_output_dir, "{basename[0]}.rmdup.bam"),
                                                     pjoin(
                                                        params.em_vc_output_dir, "{basename[0]}.rmdup.metrics"),
                                                     pjoin(
                                                        params.em_vc_output_dir, "logs",
                                                        "{basename[0]}.rmdup.log")]).follows(index_bam)

        evaluation_intervals = [x for x in zip(
            params.rqc_evaluation_intervals_names, params.rqc_evaluation_intervals)]
        coverage_stats_tasks = []
        for ev_set in evaluation_intervals:
            coverage_stats_tasks.append(
                vc_pipeline.transform(vc_pipeline_utils.coverage_stats,
                                      sorted_bam, ruffus.formatter("sort.bam"),
                                      [pjoin(params.em_vc_output_dir, "{basename[0]}." + ev_set[0] + ".coverage.metrics"),
                                       pjoin(params.em_vc_output_dir, "logs", "{basename[0]}." + ev_set[0] + ".coverage.log")],
                                      extras=[params.em_vc_genome, ev_set[1]], name=f"coverage.{ev_set[0]}").follows(index_bam))

        output_hdf_file = pjoin(params.em_vc_output_dir, '.'.join(
            (params.em_vc_basename, "cvg_metrics", "h5")))

        if 'rqc_coverage_intervals_table' in vars(params).keys():
            coverage_intervals_table = pd.read_csv(
                params.rqc_coverage_intervals_table, sep="\t", index_col=0)
            coverage_intervals_table['file'] = \
                coverage_intervals_table['file'].apply(
                    lambda x: pjoin(params.rqc_coverage_intervals_location, x))
            coverage_intervals_files = list(coverage_intervals_table['file'])

            memory = virtual_memory()
            max_jobs = memory.total // 10e9
            coverage_categories = vc_pipeline.product(vc_pipeline_utils.coverage_stats,
                                                      sorted_bam, ruffus.formatter(
                                                          "sort.bam"),
                                                      coverage_intervals_files, ruffus.formatter(
                                                          "interval_list"),
                                                      [pjoin(params.em_vc_output_dir,
                                                             "{basename[0][0]}.{basename[1][0]}.metrics"),
                                                       pjoin(params.em_vc_output_dir,
                                                             "logs", "{basename[0][0]}.{basename[1][0]}.coverage.log")],
                                                      extras=[
                                                          params.em_vc_genome, None],
                                                      name="multiple_stats")\
                .follows(index_bam).jobs_limit(max_jobs, "coverages")

            combine_coverage_categories = vc_pipeline.merge(vc_pipeline_utils.combine_coverage_metrics,
                                                            coverage_categories, output_hdf_file,
                                                            extras=[coverage_intervals_table])
        ftrt = []
        if params.em_vc_rerun_all:
            ftrt += [md1, md2, aln]
        vc_pipeline.run(multiprocess=params.em_vc_number_of_cpus,
                        logger=logger, verbose=2, forcedtorun_tasks=ftrt)

        mark_duplicates_metrics_file = (
            mark_duplicates_bam._get_output_files(True, []))
        mark_duplicates_metrics_file = vc_pipeline_utils.flatten(
            mark_duplicates_metrics_file)
        mark_duplicates_metrics_file = [
            x for x in mark_duplicates_metrics_file if x.endswith('rmdup.metrics')][0]

        md_metric = vc_pipeline_utils.parse_md_file(
            mark_duplicates_metrics_file)

        total_reads_file = count_reads._get_output_files(True, [])
        total_reads_file = vc_pipeline_utils.flatten(total_reads_file)
        total_reads_file = [x for x in total_reads_file if x.endswith("read_count.txt")][
            0]

        total_n_reads = [int(x)
                         for x in open(total_reads_file) if x.strip()][0]

        cvg_metrics_files = vc_pipeline_utils.flatten(
            [(x._get_output_files(True, [])) for x in coverage_stats_tasks])
        cvg_metrics_files = [
            x for x in cvg_metrics_files if x.endswith('coverage.metrics')]

        cvg_metrics = [vc_pipeline_utils.parse_cvg_metrics(
            x) for x in cvg_metrics_files]

        outputs = [vc_pipeline_utils.generate_rqc_output(
            md_metric, x[0], x[1], total_n_reads) for x in cvg_metrics]
        summary_df = pd.concat([x[0] for x in outputs], axis=1)
        summary_df.columns = [x[0] for x in evaluation_intervals]
        summary_df.to_hdf(output_hdf_file, key="cvg_metrics")
        for i, c in enumerate(outputs):
            c[1].to_hdf(output_hdf_file, key=f'{evaluation_intervals[i][0]}_histogram')

        print("RapidQC run: success", file=output_log, flush=True)
    except Exception as err:
        exc_info = sys.exc_info()
        print(*exc_info, file=output_log, flush=True)
        print("RapidQC run: failed", file=output_log, flush=True)
        raise(err)
