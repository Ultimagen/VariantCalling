import pandas as pd
import json
from os.path import join as pjoin
import os
import warnings
warnings.filterwarnings('ignore')

def collect_samples_details(mds):
    """
    collect samples details for UGDVVariantCallingPipeline outputs
    Parameters
    ----------
    mds - list of metadata json files of UGDV pipeline

    Returns
    -------
    dataframe with the followig columns : ['index', 'gt_sample_name', 'workflowName', 'input_cram', 'input_cram_index', 'sorter_stats_csv','sorter_stats_json']
    """
    df_sample_details = pd.DataFrame(columns=['index', 'gt_sample_name', 'workflowName', 'input_cram', 'input_cram_index', 'sorter_stats_csv',
             'sorter_stats_json'])
    i = 0
    workflowName_arr = []
    for m in mds:
        cmd = 'gsutil cp ' + m +' .'
        os.system(cmd)
        md = json.load(open('metadata.json'))

        workflowName = md['workflowName']
        workflowName_arr.append(workflowName)
        if workflowName == 'UGDVVariantCallingPipeline':
            gt_sample_name = md['outputs']["UGDVVariantCallingPipeline.gt_sample_name"]
            input_cram = md['inputs']['input_cram_bam_list'][0]
            input_cram_index = md['inputs']['input_cram_bam_index_list'][0]
            sorter_stats_csv = md['inputs']['sorter_stats_csv']
            sorter_stats_json = md['inputs']['sorter_stats_json']

            row = {'index': i,
                   'workflowName': workflowName,
                   'gt_sample_name': gt_sample_name,
                   'input_cram': input_cram,
                   'input_cram_index': input_cram_index,
                   'sorter_stats_csv': sorter_stats_csv,
                   'sorter_stats_json': sorter_stats_json}
            df_sample_details = pd.concat([df_sample_details, pd.DataFrame.from_dict([row])])
        cmd = 'rm -f metadata.json'
        os.system(cmd)
        i = i + 1
    return df_sample_details


def prepare_json_templates_for_ssqc(df_sample_details,num_samples_per_gt,json_ssqc_template,out_dir):
    """
    prepare json templaets for singleSampleQC pipeline per sample
    Parameters
    ----------
    df_sample_details - dataframe with the following columns:  ['index', 'gt_sample_name', 'workflowName', 'input_cram', 'input_cram_index', 'sorter_stats_csv','sorter_stats_json']
    num_samples_per_gt - number of sample per ground-truth-id to randomly select
    json_ssqc_template - singleSampleQC pipeline template json file
    out_dir - output directory

    Returns
    -------
    writes json template file per sample for SingleSampleQC run. filename will be: out_dir/<index>.<gt_sample_name>.json
    """
    df_sample_for_ssqc = pd.DataFrame(columns=['index', 'gt_sample_name', 'workflowName', 'input_cram', 'input_cram_index', 'sorter_stats_csv',
             'sorter_stats_json'])

    for sample in df_sample_details['gt_sample_name'].value_counts().index:
        df = df_sample_details[df_sample_details['gt_sample_name'] == sample].sample(n=num_samples_per_gt)
        df_sample_for_ssqc = pd.concat([df_sample_for_ssqc, df])

    template = json.load(open(json_ssqc_template))

    for index, row in df_sample_for_ssqc.iterrows():
        template['SingleSampleQC.agg_bam_or_cram'] = row['input_cram']
        template['SingleSampleQC.agg_bam_or_cram_index'] = row['input_cram_index']
        template['SingleSampleQC.sorter_stats_csv'] = row['sorter_stats_csv']
        template['SingleSampleQC.sorter_stats_json'] = row['sorter_stats_json']
        template['SingleSampleQC.base_file_name'] = row['gt_sample_name']
        index = row['index']
        out_json = json.dumps(template, indent=2)
        out_json_file = pjoin(out_dir, str(index) + '.' + row['gt_sample_name'] + '.json')
        with open(out_json_file, "w") as outfile:
            outfile.write(out_json)

