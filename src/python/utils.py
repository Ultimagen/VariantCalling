import numpy as np
import itertools

def revcomp(seq: str) -> str:
    '''Reverse complements DNA given as string

    Parameters
    ----------
    seq: str
        DNA string

    Returns
    -------
    str
    '''
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A',
                  'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}
    if type(seq) == str:
        reverse_complement = "".join(complement.get(base, base)
                                 for base in reversed(seq))
    elif type(seq) == list : 
        reverse_complement =  [ complement.get(base, base) for base in reversed(seq)]
    elif type(seq) == np.ndarray : 
        reverse_complement =  np.array([ complement.get(base, base) for base in reversed(seq)])

    return reverse_complement


def runs_of_one(array, axis=None):
    '''
    returns start and end (half open) of intervals of ones in a binary vector
    if axis=None - runs are identified according to the (only) non-singleton axis
    '''
    array = array.astype(np.int8)
    if isinstance(array, np.ndarray):
        array = np.array(array)
    if not axis:
        sh = [x for x in array.shape if x != 1]
        if len(sh) != 1:
            raise RuntimeError('runs_of_one - too many non-singleton axes in array')
        else:
            array = np.squeeze(array).reshape(1, -1)
            axis = 1
    if axis != 1:
        array.reshape(array.shape[::-1])
    runs_of_ones = []
    for i in range(array.shape[0]):
        one_line = array[i, :]

        diffs = np.diff(one_line)

        starts = np.nonzero(diffs == 1)[0] + 1
        if one_line[0] == 1:
            starts = np.concatenate(([0], starts))
        ends = np.nonzero(diffs == -1)[0] + 1
        if one_line[-1] == 1:
            ends = np.concatenate((ends, [len(one_line)]))

        runs_of_ones.append(zip(starts, ends))

    return runs_of_ones


def depth2bed(depth_file: str, output_bed_file: str, cutoff: int) -> None:
    '''Generates bed file of regions with coverage above certain cutoff

    Parameters
    ----------
    depth_file: str
        Name of a file that is output of `samtools depth`
    output_bed_file: str
        Name of the output file
    cutoff: int
        Minimal depth to be included

    Returns
    -------
    None
        Writes into `output_bed_file`
    '''
    iter1 = map(lambda x: x.split(), open(depth_file))
    iter2 = map(lambda x: (x[0], int(x[1]), int(x[2])), iter1)
    with open(output_bed_file, 'w') as out:
        for chroms in itertools.groupby(iter2, lambda x: x[0]):
            pos_depths = [x[1:] for x in chroms[1]]
            poss = [x[0] for x in pos_depths]
            depths = [x[1] for x in pos_depths]
            chrpos = np.zeros(max(poss) + 1)
            chrpos[poss] = depths
            chrpos = chrpos >= cutoff
            intervals = runs_of_one(chrpos)[0]
            for interval in intervals:
                out.write("{}\t{}\t{}\n".format(chroms[0], interval[0], interval[1]))


def read_genomecov_vector(bg_file, chrom, start, end):
    result = np.zeros(end - start)
    for line in open(bg_file):
        lsp = line.strip().split()
        if lsp[0] != chrom:
            continue
        else:
            st = int(lsp[1]) - start
            en = int(lsp[2]) - start
            if st >= 0 and en <= end - start:
                result[st:en] = int(lsp[3])
    return result

def searchsorted2d(a: np.ndarray ,b: np.ndarray) -> np.ndarray:
	'''
	Inserts ith element of b into sorted ith row of a

	Parameters
	----------
	a: np.ndarray
		rxc matrix, each rows is sorted
	b: np.ndarray
		rx1 vector 

	Returns
	-------
	np.ndarray
		rx1 vector of locations 
	'''
	m,n = a.shape
	b = b.ravel()
	assert b.shape[0] == a.shape[0], "Number of values of b equal number of rows of a"
	max_num = np.maximum(a.max() - a.min(), b.max() - b.min()) + 1
	r = max_num*np.arange(a.shape[0])
	p = np.searchsorted( ((a.T+r).T).ravel(), b+r )
	return p - n*np.arange(m)

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)

def shiftarray(arr: np.ndarray, num: int, fill_value: np.float=np.nan) -> np.ndarray:
    '''Shifts array by num to the right

    Parameters
    ----------
    arr: np.ndarray
        Array to be shifted
    num: int
        Shift size (negative - left shift)
    fill_value: np.float
        Fill value
    '''
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def hmer_length(seq: str, start_point: int) -> int:
    '''Return length of hmer starting at point start_point
    
    Parameters
    ---------- 
    seq: str
        Sequence
    start_point: int
        Starting point

    Returns
    -------
    int
        Length of hmer (at least 1)
    '''

    idx = start_point 
    while seq[idx].seq.upper()==seq[start_point].seq.upper():
        idx+=1
    return idx - start_point

def get_chr_sizes( sizes_file: str) -> dict : 
    '''Returns dictionary from chromosome name to size

    Parameters
    ----------
    sizes_file: str
        .sizes file (use e.g.  cut -f1,2 Homo_sapiens_assembly19.fasta.fai > Homo_sapiens_assembly19.fasta.sizes to generate)

    Returns
    -------
    dict:
        Dictionary from name to size
    '''

    return dict([ x.strip().split() for x in open(sizes_file)])

def max_merits(specificity,recall):
    '''Finds ROC envelope from multiple sets of specificity and recall
    '''
    N = specificity.shape[0]
    ind_max = np.ones(N,np.bool)
    for j in range(N):
        for i in range(N):
            if ((specificity[i]>specificity[j]) & (recall[i]>recall[j])):
                ind_max[j] = False
                continue
    ind = np.where(ind_max)[0]
    a = np.argsort(recall[ind])
    return ind[a]