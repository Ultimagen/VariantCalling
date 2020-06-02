"""Summary

Attributes
----------
array_repr : TYPE
    Description
ERROR_PROBS : str
    Description
"""
import pandas as pd
import numpy as np
from typing import Optional, Union
from collections import Counter
import itertools
import re
import sys
import boto3
import tqdm
import subprocess
import os
import pysam
import tempfile

ERROR_PROBS = "/home/ilya/proj/VariantCalling/work/190628/probability.csv"


def get_matrix(testmatrices: np.ndarray, idx: int) -> np.ndarray:
    """Summary

    Parameters
    ----------
    testmatrices : np.ndarray
        Description
    idx : int
        Description

    Returns
    -------
    np.ndarray
        Description
    """
    if len(testmatrices.shape)==4:
        return testmatrices[idx, 0, :, :].T
    else: 
        return testmatrices[idx, :, :].T


def save_tmp_kr_matrix(tensor_name: str, n_classes: int, n_flows: int, output_dir: str) -> str:
    """extract kr npy matrix from the probability tensor sequence

    Parameters
    ----------
    tensor_name : str
        File with a probability tensor
    n_classes : int
        number of classes (in case tensort sequence is not npy)
    n_flows : int
        number of flows (in case tensort sequence is not npy)
    output_dir: str
        Location to save the kr file

    Returns
    -------
    Temporary file with Kr
    """

    if not tensor_name.endswith("npy"):
        testmatrices = np.memmap(tensor_name, dtype=np.float32)
        testmatrices = testmatrices.reshape(-1, 1, n_flows, n_classes)
    else:
        testmatrices = np.load(tensor_name, mmap_mode='r')
    kr = np.squeeze(np.argmax(testmatrices, axis=3))
    fid, tempname = tempfile.mkstemp(suffix=".npy", dir=output_dir)
    np.save(tempname, kr)
    return tempname


def get_kr(key_matrix: np.ndarray, idx: int, sf: int = 100) -> np.ndarray:
    """Summary

    Parameters
    ----------
    key_matrix : np.ndarray
        Description
    idx : int
        Description
    sf : int, optional
        Description

    Returns
    -------
    np.ndarray
        Description
    """
    return (key_matrix[idx, :] + sf // 2) // sf


def key2base(key: np.ndarray, flow_order: str) -> str:
    '''Convert key to base sequence

    Parameters
    ----------
    key : np.ndarray
        Flow key
    flow_order : str
        Flow order

    Returns
    -------
    str
        Sequence
    '''
    return ''.join([k * n for k, n in zip(flow_order, key)])


def _calculate_correct_prob(matrix: np.ndarray) -> np.ndarray:
    '''Calculates the probability to be correct call at each flow
    The probabilities of errors other than the second highest are
    discarded

    Parameters
    ----------
    matrix: np.ndarray
        Flow matrix
    '''
    kr = np.argmax(matrix, axis=0)
    high_err = np.argpartition(matrix, -2, axis=0)[-2, :]
    return matrix[kr, np.arange(matrix.shape[1])] / \
        (matrix[kr, np.arange(matrix.shape[1])] +
         matrix[high_err, np.arange(matrix.shape[1])])


def _generate_quality(seq: str, kr: np.ndarray = None, correct_prob: np.ndarray = None) -> str:
    """Summary

    Parameters
    ----------
    seq : str
        Sequence
    kr : np.ndarray
        Kr (optional)
    correct_prob: np.ndarray
        probability (optional)

    Returns
    -------
    str
        I of the length of the sequence
    """

    if kr is None:
        return 'I' * len(seq)
    else:

        error_prob = 1 - correct_prob
        probs = error_prob[kr > 0]
        tmp_kr = kr[kr > 0]
        ends = np.cumsum(tmp_kr) - 1
        starts = ends + 1 - tmp_kr
        probs = probs / 2
        output = np.zeros((tmp_kr.sum()))
        output[ends] = output[ends] + probs
        output[starts] = output[starts] + probs
        output = np.clip(output, 0.0001, None)
        error_phred = (-10 * np.log10(output)).astype(np.int8) + 33
        quals = error_phred.view('c').tostring().decode()
        return quals


def write_sequences(tensor_name: str, seq_file_name: str, n_flows: int, n_classes: int, flow_order: str) -> int:
    '''Write convert the prob. tensor to sequences and write them to the file

    Parameters
    ----------
    tensor_name : str
        Name of the probability tensor
    seq_file_name : str
        Name of the file to write the sequences
    n_flows : int
        Number of flows
    n_classes : int
        Number of classes
    flow_order: str
        Flow cycle

    Returns
    ------------------
    int
        Number of records written
    '''
    if not tensor_name.endswith("npy"):
        testmatrices = np.memmap(tensor_name, dtype=np.float32)
        testmatrices = testmatrices.reshape(-1, 1, n_flows, n_classes)
    else:
        testmatrices = np.load(tensor_name, mmap_mode='r')

    flow_order = (flow_order * n_flows)[:n_flows]
    count = 0
    with open(seq_file_name, 'w') as out:
        for i in range(testmatrices.shape[0]):
            matrix = get_matrix(testmatrices, i)

            kr = np.argmax(matrix, axis=0)
            seq = key2base(kr, flow_order)
            correct_prob = _calculate_correct_prob(matrix)
            qual = _generate_quality(seq, kr, correct_prob)
            assert len(seq)==len(qual), "Sequence of a different length with quality"
            out.write(f"{seq}\t{qual}\n")
            count +=1
    return count

def extract_tags(bamfile: str, output_file: str, tag_list: list) -> None:
    """Extracts specified tags and writes them into the TSV file

    Parameters
    ----------
    bamfile : str
        Input BAM file
   output_file : str
        Output txt file
    tag_list : list
        List of tags to extract
    """
    with open(output_file, 'w') as out:
        with pysam.AlignmentFile(bamfile, check_sq=False) as inp:
            for rec in inp:
                tags = [rec.get_tag(x, with_value_type=True) for x in tag_list]
                out.write("\t".join([f'{tag_list[i]}:{tags[i][1]}:{tags[i][0]}' for i in range(len(tags))]))
                out.write("\n")


def matrix_to_sparse(matrix: np.ndarray, kr: np.ndarray,
                     probability_threshold: float = 0, 
                     probability_sf: float = 10) -> tuple:
    """Summary

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix
    kr : np.ndarray
        regressed key
    probability_threshold : float, optional
        threshold **ratio** to report
    probability_sf: float
        Phred scaling factor (default - 10)
    Returns
    -------
    tuple
        Row, column, probability ratio normalized to the kr
    """
    max_hmer = matrix.shape[0] - 1
    probability_threshold = -probability_sf * np.log10(probability_threshold)

    tmp_matrix = matrix.copy()
    kr_clip = np.clip(kr, 0, max_hmer)
    kr_val = tmp_matrix[kr_clip, np.arange(len(kr))]
    tmp_matrix[kr_clip, np.arange(len(kr))] = 0

    row, column = np.nonzero(tmp_matrix)
    values = (tmp_matrix[row, column])

    values = np.log10(values)
    norm_value = np.log10(np.clip(kr_val[column], 1e-10, None))
    normalized_values = -probability_sf * (values - norm_value)
    normalized_values = np.clip(normalized_values, -6*probability_sf, 6*probability_sf).astype(np.int16)
    suppress = normalized_values > probability_threshold
    return row[~suppress], column[~suppress], normalized_values[~suppress]


def array_repr(x):
    return ",".join([str(y) for y in x])


def write_matrix_tags(tensor_name: str, key_name: str, output_file: str,
                      n_flows: int = 280, n_classes: int = 13, 
                      probability_threshold: float = 0.003,
                      probability_sf: float = 10) -> int:
    '''Writes probability tensor into the text file

    Parameters
    ----------
    tensor_name : str
        Name of the tensor file
    key_name : str
        Regressed key file name
    output_file : str
        Name of the output file
    n_flows : int, optional
        Number of flows (default: 280)
    n_classes : int, optional
        Number of classes called (default: 13)
    probability_threshold : float, optional
        Minimal probability to report
    probability_sf: float, optional
        Scaling factor for phred probability


    Returns
    -------
    int
        Number of sequences written
    '''

    if key_name is not None:
        if not key_name.endswith("npy"):
            key = np.memmap(key_name, dtype=np.int16).reshape((-1, n_flows))
        else:
            key = np.load(key_name, mmap_mode='r')

    else:
        key = None
    if not tensor_name.endswith("npy"):
        testmatrices = np.memmap(tensor_name, dtype=np.float32)
        testmatrices = testmatrices.reshape(-1, 1, n_flows, n_classes)
    else:
        testmatrices = np.load(tensor_name, mmap_mode='r')
        if len(testmatrices.shape)==3:
            testmatrices = testmatrices[:,np.newaxis,...]

    print(f'Read {testmatrices.shape[0]} predictions', flush=True, file=sys.stderr)
    empty = 0
    complete = 0 
    with open(output_file, 'w') as out:
        for idx in tqdm.tqdm(range(testmatrices.shape[0])):
            matrix = get_matrix(testmatrices, idx)
            if key is not None:
                kr = key[idx, :]
            else:
                kr = np.argmax(matrix, axis=0)

            kh, kf, kd = matrix_to_sparse(matrix, kr, probability_threshold, probability_sf)
            if len(kh) == 0 or len(kf) == 0 or len(kd) == 0:
                out.write("\n")
                empty+=1
                continue
            kr_str = array_repr(kr)
            kr_str = 'kr:B:C,' + kr_str
            kh_str = array_repr(kh)
            kh_str = 'kh:B:C,' + kh_str
            kf_str = array_repr(kf)
            kf_str = 'kf:B:S,' + kf_str
            kd_str = array_repr(kd)
            kd_str = 'kd:B:c,' + kd_str
            out.write('\t'.join((kr_str, kd_str, kh_str, kf_str)))
            out.write("\n")
            complete+=1
    return empty, complete


def extract_header(input_bam: str, output_sam: str) -> None:
    '''Get SAM header

    Parameters
    ----------
    input_bam : str
        Input BAM file
    output_sam : str
        Output SAM file

    No Longer Returned
    ------------------
    None
    '''
    with open(output_sam, 'w') as outfile:
        subprocess.check_call(
            ['samtools', 'view', '-H', input_bam], stdout=outfile)


def add_matrix_to_bam(input_bam: str, input_matrix: str, output_bam: str, replace_sequence_file: str=None) -> None:
    """Summary

    Parameters
    ----------
    input_bam : str
        Description
    input_matrix : str
        Description
    output_bam : str
        Description
    replace_sequence_file : str, optional
        Description
    """
    if replace_sequence_file is not None:
        rgbi_fname = output_bam + "rgbi.txt"
        extract_tags(input_bam, rgbi_fname, ['RG', 'bi', 'rq'])

    re, we = os.pipe()
    extract_header(input_bam, output_bam + ".hdr")
    p1 = subprocess.Popen(['cat', output_bam + '.hdr'], stdout=we)

    p4 = subprocess.Popen(
        ['samtools', 'view', '-b', '-o', output_bam, '-'], stdin=re)
    p1.wait()
    if replace_sequence_file is None:
        p2 = subprocess.Popen(['samtools', 'view', input_bam],
                              stdout=subprocess.PIPE)
    else:
        p2a = subprocess.Popen(['samtools', 'view', input_bam],
                               stdout=subprocess.PIPE)
        p2 = subprocess.Popen(
            ['cut', '-f1-9'], stdin=p2a.stdout, stdout=subprocess.PIPE)
        p2a.stdout.close()

    if replace_sequence_file is None:
        p3 = subprocess.Popen(['paste', '-', input_matrix],
                              stdin=p2.stdout, stdout=we)
    else:
        p3 = subprocess.Popen(['paste', '-'] + [replace_sequence_file, rgbi_fname, input_matrix],
                              stdin=p2.stdout, stdout=we)

    p2.stdout.close()
    os.close(we)

    p4.wait()
    os.unlink(output_bam + ".hdr")
    if replace_sequence_file is not None:
        os.unlink(rgbi_fname)


def read_range_bytes(obj, start, end):
    """Summary

    Parameters
    ----------
    obj : TYPE
        Description
    start : TYPE
        Description
    end : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    range_header = "bytes=%d-%d" % (start, end - 1)
    tmp = obj.get(Range=range_header)['Body'].read()
    return np.frombuffer(tmp, dtype=np.int16)


def read_key_range(obj, start_idx, end_idx, read_size=280):
    """Summary

    Parameters
    ----------
    obj : TYPE
        Description
    start_idx : TYPE
        Description
    end_idx : TYPE
        Description
    read_size : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    start = start_idx * 2 * read_size
    end = end_idx * 2 * read_size
    tmp = read_range_bytes(obj, start, end)
    return tmp.reshape(-1, read_size)


def idx_to_block(idx, block_size_in_reads):
    """Summary

    Parameters
    ----------
    idx : TYPE
        Description
    block_size_in_reads : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    return idx // block_size_in_reads


def read_in_block(idx, block_size_in_reads):
    """Summary

    Parameters
    ----------
    idx : TYPE
        Description
    block_size_in_reads : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    return idx % block_size_in_reads


def block_idx_to_block_start_end(block_idx, block_size_in_reads, total_number_of_reads=1150319344):
    """Summary

    Parameters
    ----------
    block_idx : TYPE
        Description
    block_size_in_reads : TYPE
        Description
    total_number_of_reads : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    return(block_idx * block_size_in_reads, min(total_number_of_reads, (block_idx + 1) * block_size_in_reads))


def fetch_indices(s3_url: str, n_flows: int, read_indices: np.ndarray):
    '''Fetch keys that correspond to bead indices 

    Parameters
    ----------
    s3_url : str
        Regressed key url `s3://
    n_flows : int
        Number of flows read
    read_indices : np.ndarray
        Array of indices to fetch

    Returns
    -------
    TYPE
        Description
    '''
    s3 = boto3.resource("s3")
    bucket_name = re.split(r"/+", s3_url)[1]
    key = "/".join(re.split(r"/+", s3_url)[2:])
    obj = s3.Object(bucket_name=bucket_name, key=key)
    total_number_of_reads = obj.content_length / n_flows / 2
    reads_regressed_signals = np.zeros((len(read_indices), n_flows), np.int16)
    cur_block_idx = -1
    block_size_in_reads = 1000000
    for i, read_id in enumerate(tqdm.tqdm(read_indices)):
        read_block = idx_to_block(read_id, block_size_in_reads)
        if read_block != cur_block_idx:
            start, end = block_idx_to_block_start_end(
                read_block, block_size_in_reads, total_number_of_reads)
            cur_block_idx = read_block
            block = read_key_range(obj, start, end, n_flows)
        pos_in_block = read_in_block(read_id, block_size_in_reads)
        reads_regressed_signals[i, :] = block[pos_in_block, :]
    return reads_regressed_signals


def read_error_probs(error_probs_csv: str = ERROR_PROBS, binned_by_quality: bool = False,
                     left_motif_size: int= 5, right_motif_size: int=5, n_regression_bins: int=0) -> Union[pd.DataFrame, list]:
    '''Read error probs CSV and produces data frame

    Parameters
    ----------
    error_probs_csv : str, optional
        CSV file with error probabilities. Example %s
    binned_by_quality : bool, optional
        Optional - True if the probabilities are binned by the regressed signal aof the base
    left_motif_size : int, optional
        Length of the left motif. 
    right_motif_size : int, optional
        Length of the right motif
    n_regression_bins : int, optional
        Default - 0 - number of regression bins

    Returns
    -------
    Union[pd.DataFrame, list]
    '''

    source_dataframe = pd.read_csv(error_probs_csv)
    source_dataframe['left'] = source_dataframe[
        'motif'].apply(lambda x: x[:left_motif_size])
    source_dataframe['right'] = source_dataframe[
        'motif'].apply(lambda x: x[-right_motif_size:])
    source_dataframe['middle'] = source_dataframe['motif'].apply(
        lambda x: x[left_motif_size + 1:-right_motif_size - 1])
    source_dataframe['hmer_letter'] = source_dataframe[
        'middle'].apply(lambda x: x[1])
    source_dataframe['hmer_number'] = source_dataframe[
        'middle'].apply(lambda x: x[0]).astype(np.int)
    source_dataframe.drop(['motif', 'middle'], axis=1, inplace=True)
    tups = [tuple(x) for x in source_dataframe[
        ['left', 'hmer_number', 'hmer_letter', 'right']].values]
    source_dataframe.index = pd.MultiIndex.from_tuples(
        tups, names=['left', 'hmer_number', 'hmer_letter', 'right'])

    source_dataframe.drop(
        ['left', 'right', 'hmer_letter', 'hmer_number'], axis=1, inplace=True)

    assert (n_regression_bins > 0 and binned_by_quality) or (
        n_regression_bins == 0 and not binned_by_quality), "If not binned by quality - 0 bins"

    if binned_by_quality:
        n_diffs = int(source_dataframe.shape[1] / n_regression_bins)
        column_names = [f"{x-int((n_diffs-1)/2)}_{y}" for x in range(n_diffs) for y in range(n_regression_bins)]
        source_dataframe.columns = column_names
        return source_dataframe
    else:
        source_dataframe.columns = [
            'n(-1)', 'n(0)', 'n(+1)', 'P(-1)', 'P(0)', 'P(+1)']
        return source_dataframe


def _convert_to_probs(source_dataframe: pd.DataFrame):
    '''Converts counts to probabilities

    Parameters
    ----------
    source_dataframe : pd.DataFrame
            DataFrame of counts

    Returns
    -------
    adds columns P(-1), P(0), P(+1) to convert counts (n(-1), n(0),n(+1)) to probabilities

    '''
    count_columns = [
        x for x in source_dataframe.columns if not x.startswith('P')]
    source_dataframe = source_dataframe[count_columns]
    prob_columns = ['P({})'.format(x) for x in count_columns]
    sum_counts = source_dataframe[count_columns].sum(axis=1)
    sum_counts[sum_counts == 0] = .01
    probs = source_dataframe[count_columns].multiply(1 / sum_counts, axis=0)

    probs.columns = prob_columns
    source_dataframe[prob_columns] = probs
    return source_dataframe


def marginalize_error_probs(source_dataframe: pd.DataFrame, left_drop: int = 0, right_drop: int=0) -> pd.DataFrame:
    '''Marginalize error probabilities by combining motifs sharing common suffix (left) or prefix (right)
    This function is useful for calculation of error probabilities of nucleotides that are close to the end 
    of the read

    Parameters
    ----------
    source_dataframe : pd.DataFrame
            Input DataFrame
    left_drop : int, optional
            Number of nucleotides to marginalize on in prefix of left context
    right_drop : int, optional
            Number of nucleotides to marginalize on in suffix of right context

    Returns
    -------
    pd.DataFrame
        Description
    '''
    #source_dataframe = source_dataframe.drop(0, axis=0, level='hmer_number').copy()
    assert left_drop > 0 or right_drop > 0, "No marginalization needed for these drop values"
    len_left = len(source_dataframe.index.get_level_values("left")[0])
    len_right = len(source_dataframe.index.get_level_values("right")[0])
    assert left_drop <= len_left and right_drop <= len_right, "Unable to marginalize on more nucs than exist"

    groupby_left = source_dataframe.index.get_level_values('left').str[
        left_drop:]
    if right_drop > 0:

        groupby_right = source_dataframe.index.get_level_values('right').str[
            :-right_drop]
    else:
        groupby_right = source_dataframe.index.get_level_values('right')
    groupby_length = source_dataframe.index.get_level_values('hmer_number')
    groupby_hmer = source_dataframe.index.get_level_values('hmer_letter')

    mi = pd.MultiIndex.from_arrays(
        (groupby_left, groupby_right, groupby_length, groupby_hmer))
    source_dataframe1 = source_dataframe.copy()
    source_dataframe1.index = mi
    gsource_dataframe = source_dataframe1.groupby(axis=0, level=[0, 1, 2, 3])
    source_dataframe1 = gsource_dataframe.agg(np.sum)
    #source_dataframe1 = _convert_to_probs(source_dataframe1)
    source_dataframe = source_dataframe1.reorder_levels(
        ['left', 'hmer_number', 'hmer_letter', 'right'], axis=0)
    # No need to add zero model since
    #source_dataframe = add_zero_model(source_dataframe)
    source_dataframe.sort_index(inplace=True)
    return source_dataframe


def create_marginalize_dictionary(source_dataframe: pd.DataFrame) -> dict:
    '''Creates a dictionary of all possible marginalizations of the error model

    Parameters
    ----------
    source_dataframe : pd.DataFrame
        original dataframe

    Returns
    -------
    dict
        dictionary with keys - sizes of the left and right motif

    Raises
    ------
    Exception
        Description
    '''

    marginalize_dict = {}
    len_left = len(source_dataframe.index.get_level_values("left")[0])
    len_right = len(source_dataframe.index.get_level_values("right")[0])

    marginalize_dict[(len_left, len_right)] = source_dataframe.copy()
    for i in range(len_left + 1):
        for j in range(len_right + 1):
            if (len_left - i, len_right - j) in marginalize_dict:
                continue
            elif (len_left - i + 1, len_right - j) in marginalize_dict:
                source_dataframe = marginalize_dict[
                    ((len_left - i + 1, len_right - j))]
                source_dataframe1 = marginalize_error_probs(
                    source_dataframe, 1, 0)
                marginalize_dict[(len_left - i, len_right - j)
                                 ] = source_dataframe1
            elif (len_left - i, len_right - j + 1) in marginalize_dict:
                source_dataframe = marginalize_dict[
                    ((len_left - i, len_right - j + 1))]
                source_dataframe1 = marginalize_error_probs(
                    source_dataframe, 0, 1)
                marginalize_dict[(len_left - i, len_right - j)
                                 ] = source_dataframe1
            else:
                raise Exception("Can't create {}".format(
                    (len_left - i, len_right - j)))
    return marginalize_dict


def add_zero_model(source_dataframe: pd.DataFrame) -> pd.DataFrame:
    '''Add probabilities for 0->0 and 0->1 errors.  
    Due to implementation difficulties P(0|0) and P(0|1) were not reported.
    The calculation is P(0|1) = P (1|0), P(0|0) = 1-P(1|0)

    Parameters
    ----------
    source_dataframe : pd.DataFrame
        Input dataframe

    Return
    ------
    pd.DataFrame

    Note
    ----
    This function does not work yet with the new format of the error model
    '''
    new_source_dataframe = source_dataframe.xs(
        1, axis=0, level='hmer_number').copy()
    new_source_dataframe = pd.concat((new_source_dataframe,), keys=[
                                     0], names=['hmer_number'])
    new_source_dataframe = new_source_dataframe.reorder_levels(
        ['left', 'hmer_number', 'hmer_letter', 'right'], axis=0)

    # there is no meaning for counts for now in this case
    new_source_dataframe[['n(0)', 'n(-1)', 'n(+1)']] = 0
    new_source_dataframe[['P(+1)']] = new_source_dataframe[['P(-1)']]
    new_source_dataframe[['P(-1)']] = 0
    new_source_dataframe[['P(0)']] = 1 - new_source_dataframe[['P(+1)']]
    return pd.concat((new_source_dataframe, source_dataframe))


def split_by_signal_bins(source_dataframe: pd.DataFrame) -> pd.DataFrame:
    '''Recieves dataframe with hmers binned by regression signal and splits it 
    into a list by the bin of regression signal

    Parameters
    ----------
    source_dataframe : pd.DataFrame
        Source dataframe. Column names should be of the form ?(n_nn) where n is the difference
        between the read hmer and the ground truth hmer and nn is the value of the bin

    Returns
    -------
    pd.DataFrame
    '''
    column_name_regexp = r'([A-Za-z]*)\({0,1}([\-0-9]+)_([\-0-9]+)\){0,1}'
    parsed_column_names = [
        re.search(column_name_regexp, x).groups() for x in source_dataframe.columns]
    parsed_column_names = sorted(parsed_column_names, key=(
        lambda x: (int(x[2]), x[0], int(x[1]))))
    dfs = []
    grouped_column_names = itertools.groupby(
        parsed_column_names, lambda x: (x[2]))
    for g in grouped_column_names:
        take = [x for x in g[1]]
        take_names = [x[0] + '(' * (len(x[0]) > 0) + x[1] +
                      "_" + x[2] + ')' * (len(x[0]) > 0) for x in take]
        new_name = [x[0] + '(' * (len(x[0]) > 0) + x[1] +
                    ')' * (len(x[0]) > 0) for x in take]
        new_name = [int(x) if x.isdigit() or x.startswith('-')
                    else x for x in new_name]
        df = source_dataframe[take_names]

        df.columns = new_name
        dfs.append(df.copy())
    return dfs


def convert2readGivenData(source_dataframe: pd.DataFrame) -> pd.DataFrame:
    '''Convert probabilities in P(read | data = i ) to P(read=i | data)

    Parameters
    ----------
    source_dataframe : pd.DataFrame
        source dataframe. Should contain columns with counts with integer names (-2,-1,0,1,2 etc.)
        and probability columns (P(-2), P(-1)...) that represent e.g P(R=-1|H=0) (i.e. probability of 
        deletion in read given haplotype for each row)

    Returns
    -------
    pd.DataFrame
        In each row now P(-1) now represents P(R=0 | H=-1) etc. 
    '''

    left_names = source_dataframe.index.get_level_values('left').unique()
    hmer_names = source_dataframe.index.get_level_values(
        'hmer_letter').unique()
    right_names = source_dataframe.index.get_level_values('right').unique()
    hmer_number = source_dataframe.index.get_level_values('hmer_number').max()
    hmer_number = np.arange(0, hmer_number + 2)

    bins_number = int((source_dataframe.shape[1]) / 2)

    idx = pd.MultiIndex.from_product([left_names, hmer_number, hmer_names, right_names],
                                     names=['left', 'hmer_number', 'hmer_letter', 'right']).sort_values()
    result_dataframe = pd.DataFrame(
        index=idx, columns=source_dataframe.columns)

    dest_columns = []
    for err_idx in source_dataframe.columns[:bins_number]:
        source_dataframe['dest({})'.format(err_idx)] = (
            source_dataframe.index.get_level_values('hmer_number') - err_idx).astype(np.int)
        dest_columns.append('dest({})'.format(err_idx))

    # now rearranging
    for err_idx in source_dataframe.columns[:bins_number]:
        tmp = source_dataframe[source_dataframe[
            'dest({})'.format(err_idx)] >= 0]
        tmp = tmp.reset_index()
        tmp.sort_values(['left', 'dest({})'.format(err_idx),
                         'hmer_letter', 'right'], inplace=True)
        dest_index = pd.MultiIndex.from_frame(tmp[['left', 'dest({})'.format(err_idx), 'hmer_letter', 'right']],
                                              names=['left', 'hmer_number', 'hmer_letter', 'right'])

        destination_contained = dest_index.isin(result_dataframe.index)
        dest_index = dest_index[destination_contained]
        result_dataframe.loc[dest_index, err_idx] = np.array(
            tmp[-err_idx].values)[destination_contained]
        result_dataframe.loc[dest_index, 'P({})'.format(err_idx)] = np.array(
            tmp['P({})'.format(-err_idx)].values)[destination_contained]

    source_dataframe.drop(dest_columns, axis=1, inplace=True)
    result_dataframe.dropna(how='all', inplace=True)

    return result_dataframe


class ErrorModel:
    '''Contains error model and functions to access it. The design of the class 
    is mosty for efficiency purposes

    Attributes
    ----------
    error_model : TYPE
        Description
    hashed_dict : TYPE
        Description
    _em - error model. Currently implemented as numpy array, keeps only probabilities
    _hash_dict - dictionary between hash of the index and the index in the array

    Methods
    -------
    get_hash - fetch by hash of the tuple
    get_tuple - fetch by tuple (left, hmer_number, hmer_letter, right)
    get_index - directly fetch by index in the array
    '''

    def __init__(self, error_model_file: str, n_bins: int=0):
        """Summary

        Parameters
        ----------
        error_model_file : str
            Description
        n_bins : int, optional
            Description
        """
        if n_bins > 0:
            error_models = [pd.read_hdf(
                error_model_file, key="bin_{}".format(b)) for b in range(n_bins)]
            hashed_idcs = [[hash(x) for x in em.index] for em in error_models]
            self.hashed_dict = [
                dict(zip(hashed_idx, range(len(hashed_idx)))) for hashed_idx in hashed_idcs]
            del hashed_idcs
            self.error_model = [np.array(error_model[[x for x in error_model.columns if type(x) == str and
                                                      x.startswith('P')]]) for error_model in error_models]
            for i in range(len(self.error_model)):
                self.error_model[i] = np.concatenate(
                    (self.error_model[i], np.zeros((1, self.error_model[i].shape[1]))))
            del error_models
        else:
            error_model = pd.read_hdf(
                error_model_file, key="error_model_hashed")
            hashed_idx = [hash(x) for x in error_model.index]
            self.hashed_dict = dict(zip(hashed_idx, range(len(hashed_idx))))
            del hashed_idx
            self.error_model = np.array(
                error_model[['P(-1)', 'P(0)', 'P(+1)']])
            self.error_model = np.concatenate(
                (self.error_model, np.zeros((1, self.error_model.shape[1]))))
            del error_model

    def hash2idx(self, hash_list: list, bins: Optional[np.ndarray]=None) -> list:
        """Summary

        Parameters
        ----------
        hash_list : list
            Description
        bins : Optional[np.ndarray], optional
            Description

        Returns
        -------
        list
            Description
        """
        if bins is None:
            return [self.hashed_dict.get(x, self.error_model.shape[0] - 1) for x in hash_list]
        else:
            return [self.hashed_dict[bins[i]].get(hash_list[i], self.error_model[bins[i]].shape[0] - 1) for i in range(len(bins))]

    def get_hash(self, tuple_hash: int, bin: Optional[int]=None) -> np.array:
        """Summary

        Parameters
        ----------
        tuple_hash : int
            Description
        bin : Optional[int], optional
            Description

        Returns
        -------
        np.array
            Description
        """
        if bin is None:
            hashed_idx = self.hashed_dict.get(
                tuple_hash, self.error_model.shape[0] - 1)
            return self.error_model[hashed_idx, :]
        else:
            hashed_idx = self.hashed_dict[bin].get(
                tuple_hash, self.error_model[bin].shape[0] - 1)
            return self.error_model[bin][hashed_idx, :]

    def get_tuple(self, tup: tuple, bin: Optional[int]=None) -> np.array:
        """Summary

        Parameters
        ----------
        tup : tuple
            Description
        bin : Optional[int], optional
            Description

        Returns
        -------
        np.array
            Description
        """
        if bin is None:
            hashed_idx = self.hashed_dict.get(
                hash(tup), self.error_model.shape[0] - 1)
            return self.error_model[hashed_idx, :]
        else:
            hashed_idx = self.hashed_dict[bin].get(
                hash(tup), self.error_model[bin].shape[0] - 1)
            return self.error_model[bin][hashed_idx, :]

    def get_index(self, index_list: np.ndarray, bins: Optional[np.ndarray]=None) -> np.array:
        """Summary

        Parameters
        ----------
        index_list : np.ndarray
            Description
        bins : Optional[np.ndarray], optional
            Description

        Returns
        -------
        np.array
            Description
        """
        if bins is None:
            return self.error_model[index_list, :]
        else:
            output = np.zeros((len(index_list), self.error_model[0].shape[1]))
            bins_set = Counter(bins)

            for s in bins_set:
                put = (bins == s)
                output[put, :] = self.error_model[s][index_list[put], :]
            return output
