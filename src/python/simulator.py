import numpy as np
from typing import Optional

from os.path import join as pjoin
from . import utils
import pysam
from . import readExpander

error_rates = np.logspace(.001, .01, 10)
DEFAULT_FLOW_ORDER = "TACG"
_workdir = "/home/ilya/proj/VariantCalling/work/190605/"

# We always allow for a bit of error
_CONFUSION_MATRIX = np.load(pjoin(_workdir, "confusion.npy"))
min_prob = np.min(_CONFUSION_MATRIX[_CONFUSION_MATRIX > 0])
CONFUSION_MATRIX = np.clip(_CONFUSION_MATRIX, min_prob, None)
CONFUSION_MATRIX = (CONFUSION_MATRIX.T / CONFUSION_MATRIX.sum(axis=1)).T


def generateFlowMatrix(gtr_calls: np.ndarray,
                       confusion_matrix: np.ndarray = CONFUSION_MATRIX) -> np.ndarray:
    '''
    Generates a flow matrix from the ground truth flows with low probabilities for 

    Parameters
    ----------
    gtr_calls : np.ndarray 
            Ground truth in key space
    confusion_matrix : np.ndarray
            n_hmer x n_hmer confusion matrix - rows - ground truth, columns - probabilities

    Returns
    -------
    modified_flow_matrix : np.ndarray
            Modified matrix corresponding to mutations in the read, in the mutated positions the 
            probabilities will be around 0.5 and 0.5 between the ground truth and the mutation, in other positions - they will be 
            according to the probabilities of the confusion matrix
    '''

    expected_reads = confusion_matrix[gtr_calls, :]
    cum_expected = np.cumsum(expected_reads, axis=1)
    rands = np.random.uniform(0, 1, cum_expected.shape[0])
    mutation_profile = utils.searchsorted2d(cum_expected, rands)

    # this is the mutation profile in each nucleotide
    diff = mutation_profile - gtr_calls

    # something narrow around 0 and 1
    mutations = np.nonzero(diff)[0]
    mutation_probs = np.random.beta(3, 3, size=len(mutations))
    non_mutation_probs = 1 - mutation_probs

    flow_matrix = confusion_matrix[gtr_calls, :].T
    minconf = confusion_matrix.min()
    mutation_flow_matrix = np.ones(
        (confusion_matrix.shape[0], len(mutations))) * minconf

    mutation_flow_matrix[gtr_calls[mutations],
                         range(len(mutations))] = non_mutation_probs
    mutation_flow_matrix[mutation_profile[mutations],
                         range(len(mutations))] = mutation_probs
    mutation_flow_matrix = mutation_flow_matrix / \
        mutation_flow_matrix.sum(axis=0)
    flow_matrix[:, mutations] = mutation_flow_matrix
    return flow_matrix


def getFlow2Base(flow_order: str, desired_length: int) -> np.ndarray:
    '''
    Given order of flows - what base corresponds to each flow

    Parameters
    ----------
    flow_order: str
            String of flow orders
    desired_length: int
            Needed length of flows
    Returns
    -------
    np.ndarray
            Array of bases
    '''
    flow_order = np.tile(np.array(list(flow_order)), int(
        desired_length / len(flow_order)) + 1)
    flow_order = flow_order[:desired_length]
    return flow_order


def get_rflow_order(flow_matrix: np.ndarray, flow_order: str=DEFAULT_FLOW_ORDER) -> str:
    '''Finds the flow of the first base of the reverse of the expected sequence
    Parameters
    ----------
    flow_matrix: np.ndarray
        Flow matrix
    flow_order: str
        Flow order that generated the flow matrix

    Returns
    -------
    str
        Reversed flow order
    '''
    tmp = flow_order * flow_matrix.shape[1]
    tmp = tmp[:flow_matrix.shape[1]]
    rflow = utils.revcomp(tmp)
    return rflow[:len(flow_order)]


def keySpace2BaseSpace(flow_matrix: np.ndarray, flow_order: str=DEFAULT_FLOW_ORDER) -> tuple:
    '''
    Converts from matrix of hmer probabilities to sequence (most probable)

    Parameters
    ----------
    flow_matrix: np.ndarray
            n_hmer x flow matrix of hmer probabilities
    flow_order : str
            flow order (default: TCAG)

    Returns
    -------
    tuple (str, np.ndarray, np.ndarray)
            1. String of most probable bases in the read
            2. Array of the last output base BEFORE each flow (For instance for flows 01021 the output is -1 -1 0 0 2 )
            3. Array of the call for each flow

    '''
    flow_order = getFlow2Base(flow_order, flow_matrix.shape[1])
    n_reps = np.argmax(flow_matrix, axis=0)
    n_reps_shifted = np.array([0] + list(np.argmax(flow_matrix, axis=0)))[:-1]
    flow2base = -1 + np.cumsum(n_reps_shifted)
    return ''.join(np.repeat(flow_order, n_reps)), flow2base, n_reps


def identifyMutations(flow_matrix: np.ndarray, n_mutations: int, threshold: float,
                      gtr_calls: Optional[np.ndarray] = None) -> np.ndarray:
    '''
    Identifies mutations to report in the read

    Parameters
    ----------
    flow_matrix: np.ndarray
            Modified flow matrix with mutations
    n_mutations: int
            Maximal number of mutations to return
    threshold: float
            Minimal mutation probability
    gtr_calls: np.ndarray
            In case there is a ground truth - calls will be calculated relative to the ground truth

    Returns
    -------
    tuple:
            (np.ndarray, np.ndarray) List of flows in key space and list of key values corresponding to each mutation

    Note
    ----
    All mutations that have probability of n_mutation strongest mutations are passed (potentially more than n_mutations)
    '''

    mutation_id_matrix = flow_matrix.copy()
    mutation_id_matrix[mutation_id_matrix < threshold] = 0

    if gtr_calls is None:
        gtr_calls = np.argmax(flow_matrix, axis=0)

    # do not identify ground truth calls
    mutation_id_matrix[gtr_calls, np.arange(flow_matrix.shape[1])] = 0

    strongest_candidates = np.argsort(mutation_id_matrix, axis=None)[
        ::-1]

    # in case of several mutations with the same probability - pass all of them
    # this is needed to prevent cases of different reverse complement mutations
    cur_idx = n_mutations-1
    while mutation_id_matrix.flat[strongest_candidates[cur_idx]] > 0 and \
        mutation_id_matrix.flat[strongest_candidates[cur_idx]]==mutation_id_matrix.flat[strongest_candidates[n_mutations-1]]:
        cur_idx += 1
    n_mutations = cur_idx
    strongest_candidates = strongest_candidates[:n_mutations]
    # convert flat index that argsort returns to rows and columns
    take = mutation_id_matrix.flat[strongest_candidates] > 0
    tmp = np.unravel_index(
        strongest_candidates[take], mutation_id_matrix.shape)
    return tmp[1], tmp[0]


def getLikelihood(flow_matrix: np.ndarray, gtr_calls: Optional[np.ndarray] = None) -> float:
    '''
    Return log likelihood of the path in flow matrix

    Parameters
    ----------
    flow_matrix: np.ndarray
            n_hmers x n_flows matrix of hmer call probabilities
    gtr_calls: Optional[np.ndarray]
            None or specific path in flows to return likelihood. If none - optimal path is returned

    Returns
    -------
    Log2 of the likelihood of the flow
    '''

    lflow = np.log2(flow_matrix)
    if gtr_calls is not None:
        likelihood = lflow[gtr_calls, np.arange(lflow.shape[1])].sum()
    else:
        likelihood = np.max(lflow, axis=0).sum()
    return likelihood


def getMutationChange(gtr_calls: np.ndarray, mutation: tuple, baseCalls: str,
                      key2base: np.ndarray, flow_order: str = DEFAULT_FLOW_ORDER) -> tuple:
    '''
    Return the sequence that the mutation changes

    Parameters
    ----------
    gtr_calls: np.ndarray
            List of original (unmutated hmers)
    mutation: tuple
            (int, int) - flow index, new hmer
    baseCalls: str
            ground truth read
    key2base: np.ndarray
            For each flow - what was the last generated base BEFORE the call
    flow_order: str
            Flow order

    Returns
    -------
    tuple : (int, str, str)
            Pos, old sequence, new sequence
    '''

    flow_order = ''.join(getFlow2Base(flow_order, len(gtr_calls)))

    a = gtr_calls[mutation[0]]
    b = mutation[1]
    diff = a - b
    assert diff != 0, "No mutation in this position"
    seq_base = key2base[mutation[0]]
    if diff > 0:
        # DELETION
        if seq_base >= 0:
            start_pos = seq_base
            start_seq = baseCalls[start_pos:start_pos + diff + 1]
            end_seq = baseCalls[start_pos]
        elif seq_base < 0:
            # If we are deleting the first bases in the key
            start_pos = 0
            start_seq = baseCalls[start_pos:start_pos + diff + 1]
            end_seq = baseCalls[start_pos + diff]

    elif diff < 0:
        # INSERTION
        start_pos = seq_base
        if seq_base >= 0:
            start_seq = baseCalls[start_pos: start_pos + 1]
            end_seq = baseCalls[start_pos:start_pos + 1] + \
                flow_order[mutation[0]] * (-diff)
        elif seq_base == -1:
            start_pos = 0
            start_seq = baseCalls[start_pos: start_pos + 1]
            end_seq = flow_order[mutation[0]] * \
                (-diff) + baseCalls[start_pos:start_pos + 1]
    assert start_pos >= 0, "Assertion failed - start_pos < 0"
    assert len(start_seq) != len(
        end_seq), "Assertion failed - no difference b/w alleles"
    return start_pos, start_seq, end_seq


def getLogLikDiff(flow_matrix: np.ndarray, gtr_calls: np.ndarray, mutation: tuple):
    '''
    Reurns difference in log likelihood of reads

    Parameters
    ----------
    flow_matrix: np.ndarray
            n_hmer x n_flows matrix
    gtr_calls: np.ndarray
            Calls in key space
    mutation: tuple
            int - number of flow, int - new hmer
    '''
    lflow = np.log2(flow_matrix)
    original = lflow[gtr_calls[mutation[0]], mutation[0]]
    new = lflow[mutation[1], mutation[0]]
    return new - original


def seqToRecord(seq: str, rname: str,
                llk: float = Optional[float],
                alts: list = None,
                ralts: list = None) -> pysam.AlignedSegment:
    '''Convert string to pysam record

    Parameters
    ----------
    seq: str
        Read sequence
    rname: str
        Read name
    llk: Optional[float]
        Log likelihood of the sequence
    alts: list
        List of tuples (pos, ref, alt, diff)
    ralts: list
        Same as alts, but for reverse complement sequence
    Returns
    -------
    pysam.AlignedSegment
         unaligned BAM record
    '''

    seg = pysam.AlignedSegment()

    seg.query_sequence = seq
    seg.is_unmapped = True
    seg.query_name = rname
    seg.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
    seg.reference_name = "*"
    seg.next_reference_name = "*"
    seg.reference_start = 0
    seg.mapping_quality = 255
    seg.cigarstring = "*"

    def alt2str(x):
        return "%d,%s,%s,%.2f" % x

    alts = sorted(alts, key=lambda x: (x[3], x[0]))
    # note different sorting order for ralts so that it is the same as alts when we reverse
    ralts = sorted(ralts, key=lambda x: (x[3], -x[0]))
    rseq = utils.revcomp(seq)

    for a in alts:
        assert validate_alt(seq, a), "Weird mutation sequence"
    for a in ralts:
        assert validate_alt(rseq, a), "Weird rmutation sequence"
    valts = [validate_rcalt_v2(*x, seq, rseq) for x in zip(alts, ralts)]
    for a in valts:
        assert a, "Weird rvariant"
    alts = sorted(alts, key=lambda x: x[0])
    ralts = sorted(ralts, key=lambda x: (x[0]))

    alts = ";".join([alt2str(x) for x in alts])
    ralts = ";".join([alt2str(x) for x in ralts])
    seg.tags = ([('LL', llk), ("AL", alts), ("RA", ralts)])

    return seg


def validate_rcalt_v2(alt: tuple, ralt: tuple, seq: str, rseq: str) -> bool:
    '''Check that the read after application of ralt is the same as after applying alt

    Parameters
    ----------
    alt: tuple
        Alternative allele
    ralt: tuple
        Revcomp
    seq: str
        Sequence
    rseq: str
        Revcomp
    '''
    if apply_variant(rseq, ralt) == utils.revcomp(apply_variant(seq, alt)):
        return True
    return False


def apply_variant(seq: str, variant: tuple) -> str:
    '''Apply variant to sequence
    '''
    return readExpander.ReadExpander.get_read_variant_str(seq, [variant])


def validate_alt(seq: str, alt: tuple) -> bool:
    '''Check that the reference alleles are right

    Parameters
    ----------
    seq: str
        Sequence of the read
    alt: tuple
        Mutation

    Returns
    bool
    '''
    pos = alt[0]
    if pos < 0 or pos >= len(seq):
        return False
    if seq[pos:pos + len(alt[1])] == alt[1]:
        return True
    return False
