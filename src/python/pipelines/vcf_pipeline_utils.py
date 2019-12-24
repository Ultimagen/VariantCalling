import subprocess
import gzip 
import pandas as pd
import pysam
import numpy as np 
import tqdm 
import python.vcftools as vcftools
import python.utils as utils
from os.path import exists
from collections import defaultdict
from typing import Callable
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing, model_selection
from sklearn import metrics

def combine_vcf(n_parts: int, input_prefix: str, output_fname: str):
    '''Combines VCF in parts from GATK and indices the result
    Parameters
    ----------
    n_parts: int
        Number of VCF parts (names will be 1-based)
    input_prefix: str
        Prefix of the VCF files (including directory) 1.vcf.gz ... will be added
    output_fname: str
        Name of the output VCF
    '''
    input_files = [ f'{input_prefix}.{x}.vcf' for x in range(1,n_parts+1)] + [ f'{input_prefix}.{x}.vcf.gz' for x in range(1,n_parts+1)]
    input_files = [ x for x in input_files if exists(x)]
    cmd = ['bcftools', 'concat', '-o', output_fname, '-O', 'z'] + input_files
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    cmd = ['bcftools','index','-t', output_fname]
    subprocess.check_call(cmd)

def reheader_vcf( input_file: str, new_header: str, output_file: str) : 
    '''Run bcftools reheader and index

    Parameters
    ----------
    input_file: str
        Input file name
    new_header: str
        Name of the new header 
    output_file: str
        Name of the output file

    Returns
    -------
    None, generates `output_file`
    '''

    cmd = ['bcftools','reheader', '-h',new_header, input_file]
    with open(output_file, "wb") as out : 
        subprocess.check_call(cmd, stdout=out)
    cmd =['bcftools', 'index','-t',output_file]
    subprocess.check_call(cmd)


def run_genotype_concordance ( input_file: str, truth_file: str, output_prefix: str, 
    comparison_intervals: str,input_sample: str='NA12878', truth_sample='HG001'): 
    '''Run GenotypeConcordance, correct the bug and reindex

    Parameters
    ----------
    input_file: str
        Our variant calls
    truth_file: str
        GIAB (or other source truth file)
    output_prefix: str
        Output prefix
    comparison_intervals: str
        Picard intervals file to make the comparisons on
    input_sample: str
        Name of the sample in our input_file
    truth_samle: str
        Name of the sample in the truth file
    
    Returns
    -------
    None
    '''

    cmd = ['picard','GenotypeConcordance', 'CALL_VCF={}'.format(input_file), \
            'CALL_SAMPLE={}'.format(input_sample), 'O={}'.format(output_prefix), \
            'TRUTH_VCF={}'.format(truth_file), 'INTERVALS={}'.format(comparison_intervals), \
            'TRUTH_SAMPLE={}'.format(truth_sample), 'OUTPUT_VCF=true']
    subprocess.check_call(cmd)

    cmd = ['gunzip','-f', f'{output_prefix}.genotype_concordance.vcf.gz']
    print(' '.join(cmd))
    subprocess.check_call(cmd)
    with open(f'{output_prefix}.genotype_concordance.vcf') as input_file : 
        with open(f'{output_prefix}.genotype_concordance.tmp','w') as output_file : 
            for line in input_file : 
                if line.startswith("##FORMAT=<ID=PS"):
                    print("Replacing")
                    output_file.write(line.replace("Type=Integer", "Type=String"))
                else:
                    output_file.write(line)
    cmd = ['mv', output_file.name, input_file.name]
    print(' '.join(cmd))
    subprocess.check_call(cmd)
    cmd = ['bgzip',input_file.name]
    subprocess.check_call(cmd)
    cmd = ['bcftools','index', '-tf', f'{input_file.name}.gz']
    subprocess.check_call(cmd)


def filter_bad_areas(input_file_calls: str, highconf_regions: str, runs_regions: str ):
    '''Looks at concordance only around high confidence areas and not around runs

    Parameters
    ----------
    input_file_calls: str
        Calls file
    highconf_regions: str
        High confidence regions bed
    runs_regions: str or None 
        Runs
    '''

    highconf_file_name = input_file_calls.replace("vcf.gz","highconf.vcf")
    runs_file_name = input_file_calls.replace("vcf.gz","runs.vcf")
    with open(highconf_file_name,"wb") as highconf_file : 
        cmd = ['bedtools', 'intersect', '-a', input_file_calls, '-b', highconf_regions,'-nonamecheck',
            '-wa', '-header']
        subprocess.check_call(cmd, stdout=highconf_file)


    cmd = ['bgzip','-f', highconf_file_name]
    subprocess.check_call(cmd)
    highconf_file_name += '.gz'
    cmd = ['bcftools','index','-tf', highconf_file_name]
    subprocess.check_call(cmd)

    if runs_regions is not None : 
        with open(runs_file_name, "wb") as runs_file : 
            cmd = ['bedtools', 'subtract', '-a', highconf_file_name, '-b', runs_regions,'-nonamecheck',
                '-A', '-header']
            subprocess.check_call(cmd, stdout=runs_file)

        cmd = ['bgzip','-f', runs_file_name]
        subprocess.check_call(cmd)
        runs_file_name += '.gz'
        cmd = ['bcftools','index','-tf', runs_file_name]
        subprocess.check_call(cmd)


def vcf2concordance(raw_calls_file: str, concordance_file: str, format: str = 'GC') -> pd.DataFrame: 
    '''Generates concordance dataframe

    Parameters
    ----------
    raw_calls_file :str
        File with GATK calls
    concordance_file: str
        GenotypeConcordance file
    format: str
        Either 'GC' or 'VCFEVAL' - format for the concordance_file

    Returns
    -------
    pd.DataFrame
    '''

    vf = pysam.VariantFile(concordance_file)
    if format == 'GC':
        concordance = [ ( x.chrom, x.pos, x.qual, x.ref, x.alleles, x.samples[0]['GT'], x.samples[1]['GT']) for x in vf]
    elif format == 'VCFEVAL': 
        concordance = [ ( x.chrom, x.pos, x.qual, x.ref, x.alleles, x.samples[1]['GT'], x.samples[0]['GT']) for x in vf if 'CALL' not in x.info.keys() 
                                                                                                                            or x.info['CALL']!='OUT']

    concordance = pd.DataFrame(concordance)
    concordance.columns = ['chrom','pos','qual','ref','alleles', 'gt_ultima', 'gt_ground_truth']
    concordance['indel'] = concordance['alleles'].apply(lambda x: len(set(([len(y) for y in x])))>1)
    def classify(x):
        if x['gt_ultima']==(None, None) or x['gt_ultima'] == (None,):
            return 'fn'
        elif x['gt_ground_truth']==(None, None) or x['gt_ground_truth'] == (None,):
            return 'fp'
        else:
            set_gtr = set(x['gt_ground_truth'])-set([0])
            set_ultima = set(x['gt_ultima'])-set([0])
            if set_gtr == set_ultima :             
                return 'tp'
            elif set_ultima - set_gtr : 
                return 'fp'
            else: 
                return 'fn'

    concordance['classify'] = concordance.apply(classify, axis=1)
    def classify(x):
        if x['gt_ultima']==(None, None) or x['gt_ultima'] == (None,):
            return 'fn'
        elif x['gt_ground_truth']==(None, None) or x['gt_ground_truth'] == (None,):
            return 'fp'
        elif (x['gt_ultima'] == (0,1) or x['gt_ultima'] == (1,0)) and x['gt_ground_truth'] == (1,1): 
            return 'fn'
        elif (x['gt_ground_truth'] == (0,1) or x['gt_ground_truth'] == (1,0)) and x['gt_ultima'] == (1,1): 
            return 'fp'
        else:
            return 'tp'
    concordance['classify_gt'] = concordance.apply(classify, axis=1)

    concordance.index = [(x[1]['chrom'],x[1]['pos']) for x in concordance.iterrows()]
    vf = pysam.VariantFile(raw_calls_file)
    vfi = map(lambda x: defaultdict(lambda: None, x.info.items() + x.samples[0].items() + [('QUAL',x.qual), ('CHROM', x.chrom), ('POS',x.pos)]), vf)
    columns = ['chrom','pos','qual','sor', 'as_sor','as_sorp', 'fs','vqsr_val', 'qd', 'dp']
    original = pd.DataFrame([ [ x[y.upper()] for y in columns ] for x in vfi])
    original.columns = columns
    original.index = [(x[1]['chrom'],x[1]['pos']) for x in original.iterrows()]
    if format != 'VCFEVAL': 
        original.drop('qual',axis=1,inplace=True)
    else: 
        concordance.drop('qual', axis=1, inplace=True)
    concordance = concordance.join(original.drop(['chrom','pos'], axis=1))
    return concordance    

def find_thresholds( concordance: pd.DataFrame, classify_column: str = 'classify') -> pd.DataFrame : 
    quals = np.linspace(0,2000,30)
    sors = np.linspace(0,20,80)
    results = []
    pairs = []
    for q in tqdm.tqdm_notebook(quals) : 
        for s in sors : 
            pairs.append((q,s))
            tmp = (concordance[((concordance['qual']>q)&(concordance['sor']<s)) |\
                               (concordance[classify_column]=='fn')][[classify_column, 'indel']]).copy()
            tmp1 = (concordance[((concordance['qual']<q)|(concordance['sor']>s)) &\
                               (concordance[classify_column]=='tp')][[classify_column, 'indel']]).copy()
            tmp1[classify_column]='fn'
            tmp2=pd.concat((tmp, tmp1))
            results.append(tmp2.groupby([classify_column,'indel']).size())
    results = pd.concat(results,axis=1)
    results = results.T
    results.columns = results.columns.to_flat_index()

    results[('recall', 'indel')] = results[('tp',True)]/(results[('tp',True)]+results[('fn',True)])
    results[('specificity', 'indel')] = results[('tp',True)]/(results[('tp',True)]+results[('fp',True)])
    results[('recall', 'snp')] = results[('tp',False)]/(results[('tp',False)]+results[('fn',False)])
    results[('specificity', 'snp')] = results[('tp',False)]/(results[('tp',False)]+results[('fp',False)])
    results.index=pairs
    return results

def annotate_concordance(df: pd.DataFrame, fasta: str, alnfile: str) -> pd.DataFrame: 
    '''Annotates concordance data with information about SNP/INDELs and motifs

    Parameters
    ----------
    df: pd.DataFrame
        Concordance dataframe
    fasta: str
        Indexed FASTA of the reference genome
    alnfile: str
        Alignment file
    '''

    df = vcftools.classify_indel(df)
    df = vcftools.is_hmer_indel(df, fasta)
    df = vcftools.get_motif_around(df, 5, fasta)
    df = vcftools.get_coverage( df, alnfile, 10 )
    df = vcftools.close_to_hmer_run( df, runfile, min_hmer_run_length=10, max_distance=10)
    return df

def close_to_hmer_run( df: pd.DataFrame, runfile: str, min_hmer_run_length: int=10, max_distance: int=10) -> pd.DataFrame:
    '''Adds column is_close_to_hmer_run and inside_hmer_run that is T/F'''
    df['close_to_hmer'] = False
    df['inside_hmer_run'] = False
    run_df = utils.parse_runs_file(runfile, min_hmer_run_length)
    gdf = df.groupby('chrom')
    grun_df = run_df.groupby('chromosome')
    for chrom in gdf.groups.keys() : 
        gdf_ix = gdf.groups[chrom]
        grun_ix = grun_df.groups[chrom]
        pos1 = np.array(df.loc[gdf_ix,'pos'])
        pos2 = np.array(run_df.loc[grun_ix,'start'])
        pos1_closest_pos2_start = np.searchsorted(pos2, pos1)-1
        close_dist = (pos1-pos2[np.clip(pos1_closest_pos2_start,0,None)]) < max_distance
        close_dist |= (pos2[np.clip(pos1_closest_pos2_start+1,None, len(pos2)-1)]-pos1) < max_distance
        pos2 = np.array(run_df.loc[grun_ix,'end'])
        pos1_closest_pos2_end = np.searchsorted(pos2, pos1)
        close_dist |= (pos1-pos2[np.clip(pos1_closest_pos2_end-1,0,None)]) < max_distance
        close_dist |= (pos2[np.clip(pos1_closest_pos2_end,None, len(pos2)-1)]-pos1) < max_distance
        
        is_inside = pos1_closest_pos2_start == pos1_closest_pos2_end
        df.loc[gdf_ix, "inside_hmer_run"] = is_inside
        df.loc[gdf_ix, "close_to_hmer_run"] = (close_dist & (~is_inside))
    return df

        


def get_r_s_i( results, var_type ) -> tuple:
    '''Returns data for plotting ROC curve

    Parameters
    ----------
    results: pd.DataFrame
        Output of vcf_pipeline_utils.find_threshold
    var_type: str 
        'snp' or 'indel'

    Returns
    -------
    tuple: (pd.Series, pd.Series, np.array, pd.DataFrame)
        recall of the variable, specificity of the variable, indices of rows 
        to calculate ROC curve on (output of `max_merits`) and dataframe to plot
        ROC curve
    '''

    recall = results[('recall', var_type)]
    specificity = results[('specificity', var_type)]
    idx = utils.max_merits(np.array(recall), np.array(specificity))
    results_plot = results.iloc[idx]
    return recall, specificity, idx, results_plot


FEATURES=['sor','qd','dp','qual','coverage', 'hmer_indel_nuc']
def feature_prepare( df: pd.DataFrame ) :
    '''Prepare dataframe for analysis (encode features, normalize etc.)

    Parameters
    ----------
    df: pd.DataFrame 
        Input dataframe, only features
    feature:vec: list
        List of features 
    '''

    encode = preprocessing.LabelEncoder()
    if 'hmer_indel_nuc' in df.columns : 
        df.loc[(df['hmer_indel_nuc']).isnull(),'hmer_indel_nuc']='N'
        df.loc[:, 'hmer_indel_nuc'] = encode.fit_transform(df['hmer_indel_nuc'])
    if 'qd' in df.columns : 
        df.loc[df['qd'].isnull(), 'qd'] = 0  
    return df 


def precision_recall_of_set( input_set: pd.DataFrame, gtr_column: str, model: RandomForestClassifier ): 
    '''Precision recall calculation on a subset of data 

    Parameters
    ----------
    input_set: pd.DataFram        
    gtr_column: str
        Name of the column to calculate precision-recall on 
    model: RandomForestClassifier
        Trained model
    '''        

    features = feature_prepare( input_set[FEATURES])
    gtr_values = input_set[gtr_column]
    fns = np.array(gtr_values=='fn')
    if features[~fns].shape[0] > 0 :
        predictions = model.predict(features[~fns])
    else: 
        predictions = np.array([])
    predictions = np.concatenate((predictions, ['fp']*fns.sum()))
    gtr_values = np.concatenate((gtr_values[~fns], ['tp']*fns.sum()))
    recall = metrics.recall_score(gtr_values, predictions, pos_label='tp')
    precision = metrics.precision_score(gtr_values, predictions, pos_label='tp')
    return (precision, recall)

def calculate_precision_recall(concordance: pd.DataFrame, model: RandomForestClassifier, 
    test_train_split: pd.Series, selection: Callable, gtr_column: str) -> pd.Series : 
    '''Calculates precision and recall on a model trained on a subset of data

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    model: RandomForestClassifier
        Trained classifier
    test_train_split: pd.Series
        Split between the training and the testing set (boolean, 1 is train)
    selection: pd.Series
        Boolean series that mark specific selected rows in the concordance datafraeme
    gtr_column: str
        Name of the column 

    Returns
    -------
    pd.Series:
        The following fields are defined: 
        ```
        Three parameters for unfiltered data
        basic_recall
        basic_precision
        basic_counts 
        Four parameters for the filtered data
        train_precision
        train_recall
        test_precision
        test_recall
        ```
    '''

    tmp = concordance[selection(concordance)][gtr_column].value_counts()
    basic_recall = tmp.get('tp',0)/(tmp.get('tp',0) + tmp.get('fn',0)+1)
    basic_precision = tmp.get('tp',0)/(tmp.get('tp',0) + tmp.get('fp',0)+1)
    basic_counts  = tmp.get('tp',0) + tmp.get('fn',0)

    test_set = concordance[ selection(concordance) & (~test_train_split) ]
    test_set_precision_recall = precision_recall_of_set( test_set, gtr_column, model)

    train_set = concordance[ selection(concordance) & test_train_split ]
    train_set_precision_recall = precision_recall_of_set( train_set, gtr_column, model)

    return pd.Series((basic_recall, basic_precision, basic_counts) 
        + train_set_precision_recall + test_set_precision_recall, 
        index=['basic_recall','basic_precision', 'basic_counts',
        'train_precision','train_recall','test_precision', 'test_recall'])

def train_model( concordance: pd.DataFrame, test_train_split: np.ndarray, 
    selection: pd.Series, gtr_column: str) -> RandomForestClassifier:
    '''Trains model on a subset of dataframe that is already dividied into a testing and training set

    Parameters
    ----------
    concordance: pd.DataFrame
        Concordance dataframe
    test_train_split: pd.Series or np.ndarray
        Boolean array, 1 is train
    selection: pd.Series
        Boolean series that points to data selected for the model
    gtr_column: str
        Column with labeling

    Returns
    -------
    RandomForestClassifier
        Trained classifier model
    '''
    fns = np.array(concordance[gtr_column]=='fn')
    train_data = concordance[test_train_split & selection & (~fns)][FEATURES]
    labels = concordance[test_train_split & selection & (~fns)][gtr_column]
    train_data = feature_prepare(train_data)
    model = RandomForestClassifier(max_depth=5, class_weight={'tp':1.1, 'fp':1})
    model.fit(train_data, labels)
    return model

def train_models( concordance: pd.DataFrame, gtr_column: str, 
    selection_functions = [ lambda x: np.ones(x.shape[0], dtype=np.bool)] ) -> tuple:
    test_train_split = np.random.uniform(0,1,size=concordance.shape[0])>0.5
    selections = [] 
    models = [] 
    for sf in selection_functions : 
        selection = sf(concordance)
        model = train_model(concordance, test_train_split, selection, gtr_column)

        models.append(model)

    return test_train_split, models

def get_training_selection_functions() : 
    sfs = [ ]
    sfs.append(lambda x: ~x.indel)
    sfs.append(lambda x : x.indel & (x.hmer_indel_length == 0 ))
    sfs.append(lambda x: x.indel & (x.hmer_indel_length > 0 ))
    return sfs

def get_testing_selection_functions() : 
    sfs = [ ]
    sfs.append((lambda x: ~x.indel, 'SNP'))
    sfs.append((lambda x: x.indel, "INDEL"))
    sfs.append((lambda x : x.indel & (x.hmer_indel_length == 0 ), "Non-hmer INDEL"))
    sfs.append((lambda x: x.indel & (x.hmer_indel_length > 0 ) & (x.hmer_indel_length < 5), "HMER indel < 4"))
    sfs.append((lambda x: x.indel & (x.hmer_indel_length >= 5 ) & (x.hmer_indel_length < 12), "HMER indel > 4, < 12"))
    sfs.append((lambda x: x.indel & (x.hmer_indel_length >= 12), "HMER indel > 12"))
    return sfs
