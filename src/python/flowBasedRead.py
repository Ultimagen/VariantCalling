# Flow-based read/haplotype class
# This will be a class that would hold a read in flow base
from __future__ import annotations
from . import utils
from . import simulator
from . import error_model
import numpy as np
import pysam
import BeadsData
from typing import Optional
import copy
from collections import Counter

DEFAULT_ERROR_MODEL_FN = "/home/ilya/proj/VariantCalling/work/190628/error_model.r2d.hd5"
#DEFAULT_ERROR_MODEL = pd.read_hdf(DEFAULT_ERROR_MODEL_FN, "error_model")


def get_bam_header(rgid: Optional[str] = None):

    if rgid is None:
        rgid = '1'
    dct = {'HQ': {'VN': '0.1', 'SO': 'unsorted'},
           'RG': {'ID': rgid, 'FO': simulator.DEFAULT_FLOW_ORDER,
                  'PL': 'ULTIMA', 'SM': 'NA12878'}}

    header = pysam.AlignmentHeader(dct)
    return header


class FlowBasedRead:
    '''Class that helps working with flow based reads

    This is the class that allows for comparison between 
    reads in flow base, applying error model, outputting uBAM etc.  

    Attributes
    ----------
    read_name: str
        Read name
    seq: str
        Original read sequence
    key: np.ndarray
        sequence in flow base
    flow2base: np.ndarray
        position of the last output sequence base _before_ each flow for forward key
    flow_order: str
        sequence of flows
    _motif_size: int
        size of motif (left or right)
    _regressed_signal: np.ndarray
        Regressed signal array **binned relative to the error model**
    Methods
    -------
    apply_cigar: 
        Returns a new read with cigar applied (takes care of hard clipping and soft clipping)
    '''

    def __init__(self, dct: dict):
        '''Generic constructor

        Parameters
        ----------
        Receives key-value dictionary and sets the corresponding properties of the object

        Returns
        -------
        None
        '''

        for k in dct:
            setattr(self, k, dct[k])
        self.flow2base = self._key2base(self.key).astype(np.int)
        self.flow_order = simulator.getFlow2Base(
            self.flow_order, len(self.key))

        self._validate_seq()

    @classmethod
    def from_tuple(cls, read_name: str, read: str, error_model: error_model.ErrorModel = None,
                   flow_order: str=simulator.DEFAULT_FLOW_ORDER,
                   motif_size: int=5, max_hmer_size: int=9, ):
        '''Constructor from FASTA record and error model. Sets `seq`, `r_seq`, `key`, 
        `rkey`, `flow_order`, `r_flow_order` attributes. 

        Parameters
        ----------
        read_name: str
            Name of the read
        seq: str
            DNA sequence of the read (basecalling output)
        flow_order: np.ndarray
            Array of chars - base for each flow
        error_model: pd.DataFrame
            Error model from motif, hmer to probability that data with +1,-1,0 of that hmer generated this motif. 
            Error model could be None, in this case it will be assumed that there are no errors
        motif_size: int
            Size of the motif
        '''
        dct = {}
        dct['read_name'] = read_name
        dct['seq'] = read
        dct['forward_seq'] = read
        dct['key'] = BeadsData.BeadsData.generateKeyFromSequence(
            dct['forward_seq'], flow_order = flow_order  )
        dct['flow_order'] = flow_order
        dct['_error_model'] = error_model
        dct['_max_hmer'] = max_hmer_size
        dct['_motif_size'] = motif_size
        dct['direction'] = 'synthesis'
        return cls(dct)

    @classmethod
    def from_sam_record(cls, sam_record: pysam.AlignedSegment,
                        error_model: Optional[error_model.ErrorModel] = None,
                        flow_order: str=simulator.DEFAULT_FLOW_ORDER,
                        motif_size: int=5, max_hmer_size: int=9, format: str='ilya'):
        '''Constructor from BAM record and error model. Sets `seq`, `r_seq`, `key`, 
        `rkey`, `flow_order`, `r_flow_order` and `_flow_matrix` attributes

        Parameters
        ----------
        read_name: str
            Name of the read
        seq: str
            DNA sequence of the read (basecalling output)
        flow_order: np.ndarray
            Array of chars - base for each flow
        error_model: pd.DataFrame
            Error model from motif, hmer to probability that data with +1,-1,0 of that hmer generated this motif
            Can be optional if the read has ks, kq, kd fields
        motif_size: int
            Size of the motif
        max_hmer_size: int
            Maximal reported hmer size
        format: str
            Can be 'matt' or 'ilya'
        Returns
        -------
        Object 
        '''
        dct = {}
        dct['record'] = sam_record
        dct['read_name'] = sam_record.query_name
        dct['seq'] = sam_record.query_sequence
        dct['is_reverse'] = sam_record.is_reverse
        if sam_record.is_reverse:
            dct['forward_seq'] = utils.revcomp(dct['seq'])
        else:
            dct['forward_seq'] = dct['seq']
        dct['_error_model'] = error_model

        assert format in ['ilya','matt'], f"Format {format} not supported"
        if format == 'ilya':
            if sam_record.has_tag('ks'):
                dct['key'] = np.array(sam_record.get_tag('ks'), dtype=np.int8)
            elif sam_record.has_tag('kr'):
                dct['key'] = np.array(sam_record.get_tag('kr'), dtype=np.int8)
            else:
                dct['key'] = BeadsData.BeadsData.generateKeyFromSequence(dct[
                                                'forward_seq'], flow_order=flow_order)
        elif format == 'matt': 
            dct['key'] = np.array(sam_record.get_tag('kr'), dtype=np.int8)

        dct['_max_hmer'] = max_hmer_size
        dct['_motif_size'] = motif_size
        dct['flow_order'] = flow_order
        if format == 'ilya':
            if sam_record.has_tag("kf"):
                row = sam_record.get_tag("kh")
                col = sam_record.get_tag("kf")
                vals = sam_record.get_tag("kd")
                shape = (max_hmer_size+1, len(dct['key']))
                flow_matrix = cls._matrix_from_sparse(row, col, vals, shape)
                dct['_flow_matrix'] = flow_matrix
        elif format=='matt':
            row = np.array(sam_record.get_tag("kh"))
            col = np.array(sam_record.get_tag("kf"))
            vals = np.array(sam_record.get_tag("kd"))
            row = np.concatenate((row, dct['key'].astype(row.dtype)))
            col = np.concatenate((col, np.arange(len(dct['key'])).astype(col.dtype)))
            vals = np.concatenate((vals, np.zeros(len(dct['key']), dtype=np.float)))
            shape = (max_hmer_size+1, len(dct['key']))
            flow_matrix = cls._matrix_from_sparse(row, col, vals, shape)
            dct['_flow_matrix'] = flow_matrix

        dct['cigar'] = sam_record.cigartuples
        dct['start'] = sam_record.reference_start
        dct['end'] = sam_record.reference_end
        dct['direction'] = 'synthesis'
        return cls(dct)

    @classmethod
    def from_sam_record_rsig(cls, sam_record: pysam.AlignedSegment, regressed_signal = np.ndarray, 
                        error_model: Optional[error_model.ErrorModel] = None,
                        flow_order: str=simulator.DEFAULT_FLOW_ORDER,
                        motif_size: int=5, max_hmer_size: int=9):
        '''Constructor from BAM record and error model. Sets `seq`, `r_seq`, `key`, 
        `rkey`, `flow_order`, `r_flow_order` and `_flow_matrix` attributes

        Parameters
        ----------
        read_name: str
            Name of the read
        seq: str
            DNA sequence of the read (basecalling output)
        flow_order: np.ndarray
            Array of chars - base for each flow
        regressed_signal: np.ndarray
            Array of regressed_signals
        error_model: pd.DataFrame
            Error model from motif, hmer to probability that data with +1,-1,0 of that hmer generated this motif
            Can be optional if the read has ks, kq, kd fields
        motif_size: int
            Size of the motif
        max_hmer_size: int
            Maximal reported hmer size

        Returns
        -------
        Object 
        '''
        dct = vars(cls.from_sam_record(sam_record, error_model, flow_order, motif_size, max_hmer_size))

        dct['_regressed_signal'] = regressed_signal[:len(dct['key'])]
        return cls(dct)

    @classmethod
    def from_sam_record_flow_matrix(cls, sam_record: pysam.AlignedSegment, flow_matrix = np.ndarray, 
                        flow_order: str=simulator.DEFAULT_FLOW_ORDER,
                        motif_size: int=5, max_hmer_size: int=9):
        '''Constructor from BAM record and error model. Sets `seq`, `r_seq`, `key`, 
        `rkey`, `flow_order`, `r_flow_order` and `_flow_matrix` attributes

        Parameters
        ----------
        read_name: str
            Name of the read
        seq: str
            DNA sequence of the read (basecalling output)
        flow_order: np.ndarray
            Array of chars - base for each flow
        regressed_signal: np.ndarray
            Array of regressed_signals
        error_model: pd.DataFrame
            Error model from motif, hmer to probability that data with +1,-1,0 of that hmer generated this motif
            Can be optional if the read has ks, kq, kd fields
        motif_size: int
            Size of the motif
        max_hmer_size: int
            Maximal reported hmer size

        Returns
        -------
        Object 
        '''
        dct = vars(cls.from_sam_record(sam_record, None, flow_order, motif_size, max_hmer_size))
        
        dct['_flow_matrix'] = flow_matrix[:, :len(dct['key'])]
        return cls(dct)


    def _validate_seq(self) -> None:
        '''Validates that there are no hmers longer than _max_hmer

        Parameters
        ----------
        None

        Returns
        -------
        None. Sets attribute _validate
        '''
        self._validate = ~np.any(self.key > (self._max_hmer - 1))

    def is_valid(self) -> bool:
        '''Returns if the key is valid '''
        return self._validate

    def read2FlowMatrix(self, regressed_signal_only: bool=False) -> tuple:
        '''Gets the hmerxflow probability matrix 
        matrix[i,j] = P(read_hmer==read_hmer[j] | data_hmer[j]==i)

        Paramters
        ---------
        regressed_signal_only: bool 
            Use only regressed_signal without motifs
        Returns 
        -------
        np.ndarray 
            Flow matrix of (max_hmer_size+1) x n_flow

        '''

        if not hasattr(self, "_flow_matrix"):
            if regressed_signal_only: 
                self._flow_matrix = self._getSingleFlowMatrixOnlyRegressed(self._regressed_signal)
                return self._flow_matrix
            if not hasattr(self, '_regressed_signal'):
                self._flow_matrix = self._getSingleFlowMatrix(
                    self.key, self.flow2base, self.forward_seq)
            else: 
                self._flow_matrix = self._getSingleFlowMatrix(
                    self.key, self.flow2base, self.forward_seq, self._regressed_signal)
        return self._flow_matrix

    def _getSingleFlowMatrixOnlyRegressed(self,regressed_signal: np.ndarray, threshold=0.01) : 
        q1 = 1/np.add.outer(regressed_signal,-np.arange(self._max_hmer+1))**2
        q1 /= q1.sum(axis=-1,keepdims=True)
        
        if threshold is not None:
            q1[q1<threshold] = 0
        q1 = self._fix_nan(q1)        
        return q1.T    


    def _getSingleFlowMatrix(self, key: np.ndarray, flow2base: np.ndarray, 
        seq: str, regressed_signal: Optional[np.ndarray] = None) -> np.ndarray:
        '''Returns matrix flow matrix for a given flow key. Note that if the error model is None 
        it is assumed that there are no errors (useful for getting flow matrix of a haplotype)

        Parameters
        ----------
        key: np.ndarray 
            Hmer for each flow
        flow2base : np.ndarray 
            For each flow - what was the last base output **before** the flow
        seq: str
            Sequence of the read
        regressed_signal: 
            
        Returns
        -------
        hmerxflow probability matrix 
        '''

        if self._error_model is None:
            flow_matrix = np.zeros((self._max_hmer + 1, len(key)))
            flow_matrix[np.clip(self.key, 0, self._max_hmer),
                        np.arange(len(self.key))] = 1
            return flow_matrix

        motifs_left = []
        for i in range(key.shape[0]):
            left_base = flow2base[i]
            if left_base < 0:
                motifs_left.append('')
            else:
                motifs_left.append(
                    seq[max(left_base - self._motif_size + 1, 0):left_base + 1])
                assert seq[left_base] != self.flow_order[
                    i], "Something wrong with motifs"
        motifs_right = []
        for i in range(key.shape[0] - 1):
            right_base = flow2base[i + 1]
            motifs_right.append(
                seq[right_base + 1:right_base + self._motif_size + 1])
            assert seq[right_base +
                       1] != self.flow_order[i], "Something wrong with motifs"
        motifs_right.append("")
        index = [x for x in zip(motifs_left, key, self.flow_order[
                                :len(key)], motifs_right)]
        hash_idx = [hash(x) for x in index]

        if hasattr(self, '_regressed_signal'): 
            bins = self._regressed_signal[:len(hash_idx)]
        else: 
            bins = None
        idx_list = np.array(self._error_model.hash2idx(hash_idx, bins))
        tmp = self._error_model.get_index(idx_list, bins)

        # index
        pos = np.arange(len(key))
        key = key
        diffs = [ key+x for x in np.arange(-(tmp.shape[1]-1)/2, (tmp.shape[1]-1)/2+1)]

        flow_matrix = np.zeros((self._max_hmer + 1, len(key)))

        a1 = np.concatenate(diffs).astype(np.int)
        p1 = np.tile(pos, tmp.shape[1])
        v1 = tmp.T.ravel()
        #print(a1[:10], p1[:10], v1[:10])
        take = (a1 >= 0) & (a1 <= self._max_hmer)

        flow_matrix[a1[take], p1[take]] = v1[take]
        flow_matrix = self._fix_nan(flow_matrix)
        return flow_matrix

    def _fix_nan(self, flow_matrix: np.ndarray) -> np.ndarray:
        ''' Fixes cases when there is nan in the flow matrix. 
        This is an ugly hack - we assume that nan come from cases when 
        there is not enough data - currently we will them by 0.9 for P(R=i|H=i) and 
        by 0.1 P(r=i+-1|H=i). The only exception is P(R=i|H=0) these missing data tend to 
        come from the fact that such an insertion would require cycle skip that we do not allow

        Parameters
        ----------
        flow_matrix: np.ndarray
            Uncorrected flow matrix

        Returns
        -------
        np.ndarray 
            Corrected matrix
        '''
        gtr = self.key
        row, col = np.nonzero(np.isnan(flow_matrix))
        take = (row == gtr[col])
        zeros = (row==0)
        flow_matrix[row[take], col[take]] = 0.9
        flow_matrix[row[~take], col[~take]] = 0.1
        # missing data from P(|H=0) is coming from cycle skips
        flow_matrix[row[zeros], col[zeros]] = 0
        return flow_matrix

    def _key2base(self, key: np.ndarray) -> np.ndarray:
        '''
        Returns an array that for every flow outputs the last base output at the beginning of the flow

        Parameters
        ----------
        None 

        Returns
        -------
        np.ndarray
                Array of the last output base BEFORE each flow (For instance for flows 01021 the output is -1 -1 0 0 2 )

        '''

        n_reps_shifted = utils.shiftarray(key, 1, 0)
        flow2base = -1 + np.cumsum(n_reps_shifted)
        return flow2base

    def _get_phred_flow(self) -> np.ndarray:
        '''
        Returns Quality of flow in Phred

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray 
            Phred scaled **integer** qualities
        '''

        tmp = np.max(self._flow_matrix, axis=0)
        high_conf = (tmp==1)
        tmp = -10 * np.log10(1 - tmp)
        tmp[high_conf] = 60
        return tmp.astype(np.int8)

    def _matrix_to_sparse(self, probability_threshold: float=0) -> tuple:
        '''Converts the flow matrix to the tag representation

        Parameters
        ----------
        probability_threshold: float
            Optional threshold that would suppress variants with ratio to the best variant below the threshold

        Returns
        -------
        tuple (np.ndarray, np.ndarray, np.ndarray): 
            row, column, values (differences in int(log10) from the maximal value)
        '''

        probability_threshold = -10*np.log10(probability_threshold)
        tmp_matrix = self._flow_matrix.copy()
        tmp_matrix[self.key[self.key <= self._max_hmer], 
                    np.arange(len(self.key))[self.key <= self._max_hmer]] = 0
        row_max = np.max(self._flow_matrix, axis=0)

        row, column = np.nonzero(tmp_matrix)
        values = (tmp_matrix[row, column])

        values = np.log10(values)
        col_max = tmp_matrix[np.clip(self.key, 0, self._max_hmer), np.arange(len(self.key)) ]
        normalized_values = -10 * (values - col_max[column])
        normalized_values = np.clip(normalized_values, -60, 60)
        
        suppress = normalized_values > probability_threshold
        
        # do not output the key itself as it is always zero
        suppress = suppress | ( self.key[column]==row )
        return row[~suppress], column[~suppress], normalized_values[~suppress]


    @classmethod
    def _matrix_from_sparse( self, row, column, values, shape ):
        flow_matrix = np.zeros(shape)
        kd = np.array(values, dtype=np.float)
        kd = -kd
        kd = kd/10
        kd = 10**(kd)
        flow_matrix[row,  column] = kd
        return flow_matrix

    def to_record(self, hdr: Optional[pysam.AlignmentHeader]=None, probability_threshold: float=0) -> pysam.AlignedSegment:
        '''Converts flowBasedRead into BAM record

        Parameters
        ----------
        hdr: pysam.AlignmentHeader
            Optional header
        probability_threshold : float
            Optional - do not report variants with probability ratio to the best variant 
            less than probability_threshold (default: 0)
        Returns
        -------
        pysam.AlignedSegment
        '''
        if hasattr(self, 'record'):
            res = self.record
        else:
            res = pysam.AlignedSegment(hdr)
            res.query_sequence = self.seq
            res.query_name = self.read_name
        res.set_tag("KS", ''.join(self.flow_order[:4]))
        if hasattr(self, "_flow_matrix"):
            alt_row, alt_col, alt_val = self._matrix_to_sparse( probability_threshold = probability_threshold)

            res.set_tag('kr', [int(x) for x in self.key])
            res.set_tag('kh', [int(x) for x in alt_row])
            res.set_tag('kf', [int(x) for x in alt_col])
            res.set_tag('kd', [int(x) for x in alt_val])
        else:
            res.set_tag('kr', [int(x) for x in self.key]) 
        self.record = res
        return res

    def _left_clipped_flows(self, cigar: Optional[list] = None) -> tuple:
        '''Returns number of flows clipped from the left

        Parameters
        ----------
        cigar: list
            Optional cigar, if None - the object's cigar will be used

        Returns
        -------
        tuple (int, int)
            (fn, n) subtract n from the hmer count of flow fn, trim all flows before fn
        '''
        if cigar is None : 
            cigar = self.cigar
        if cigar[0][0] != 5:
            return (0, 0)
        else:
            bases_clipped = cigar[0][1]
            idx = 0
            stop_clip = np.argmax(self.flow2base + self.key >= bases_clipped)
            stop_clip_flow = stop_clip
            hmer_clipped = bases_clipped - self.flow2base[stop_clip] - 1

            return (stop_clip_flow, hmer_clipped)

    def _right_clipped_flows(self, cigar: Optional[list] = None) -> tuple:
        '''Returns number of flows clipped from the right

        Parameters
        ----------
        cigar: list
            Optional cigar, if None - the object's cigar will be used

        Returns
        -------
        tuple (int, int)
            (fn, n) subtract n from the hmer count of flow -fn-1, trim all flows after -fn-1
        '''
        if cigar is None : 
            cigar = self.cigar

        if cigar[-1][0] != 5:
            return (0, 0)
        else:
            bases_clipped = cigar[-1][1]
            idx = 0
            reverse_flow2base = self._key2base(self.key[::-1])

            stop_clip = np.argmax(reverse_flow2base + self.key[::-1] >= bases_clipped)
            stop_clip_flow = stop_clip
            hmer_clipped = bases_clipped - reverse_flow2base[stop_clip] - 1
            return (stop_clip, hmer_clipped)

    def apply_alignment(self) -> FlowBasedRead:
        '''Applies alignment (inversion / hard clipping ) to the flowBasedRead

        Parameters
        ----------
        None

        Returns
        -------
        New FlowBasedRead with 
        Modifies `key`, `_flow_matrix`, `flow2base`, `flow_order` attributes
        '''

        assert(hasattr(self, 'cigar')), "Only aligned read can be modified"
        attrs_dict = vars(self)
        other = FlowBasedRead(copy.deepcopy(attrs_dict))
        if other.is_reverse and self.direction != 'reference':
            if hasattr(other, '_flow_matrix'):
                other._flow_matrix = other._flow_matrix[:, ::-1]

            other.key = other.key[::-1]
            other.flow2base = other._key2base(other.key)
            other.flow_order = utils.revcomp(other.flow_order)
            
            if hasattr(other, "_regressed_signal"):
                other._regressed_signal = other._regressed_signal[::-1]

        clip_left, left_hmer_clip = other._left_clipped_flows()
        clip_right, right_hmer_clip = other._right_clipped_flows()
        
        assert left_hmer_clip >= 0 and right_hmer_clip >= 0, "Some problem with hmer clips"
        original_length = len(other.key)
        other.key[clip_left] -= left_hmer_clip
        if hasattr(other, "_regressed_signal"):
            other._regressed_signal[clip_left] -= left_hmer_clip
        # if no flows left on the left hmer - truncate it too

        shift_left = True
        if clip_left >= 0 or left_hmer_clip >= 0 :
            while other.key[clip_left] == 0:
                clip_left += 1
                shift_left = False

        other.key[-1 - clip_right] -= right_hmer_clip
        if hasattr(other, "_regressed_signal") : 
            other._regressed_signal[-1-clip_right] -= left_hmer_clip

        shift_right = True
        if clip_right >= 0 or right_hmer_clip >= 0 :
            while other.key[-1 - clip_right] == 0:
                shift_right = False
                clip_right += 1

        other.key = other.key[clip_left:original_length - clip_right]
        other._regressed_signal = other._regressed_signal[clip_left:original_length - clip_right]
        other.flow2base = other.flow2base[clip_left:original_length - clip_right]
        other.flow_order = other.flow_order[
            clip_left:original_length - clip_right]

        if hasattr(other, '_flow_matrix'):
            other._flow_matrix = other._flow_matrix[:,
                                clip_left:original_length - clip_right]
            
            if shift_left:
                other._flow_matrix[:, 0] = utils.shiftarray(
                    other._flow_matrix[:, 0], -left_hmer_clip,0)
            if shift_right:
                other._flow_matrix[:, -1] = utils.shiftarray(
                    other._flow_matrix[:, -1], -right_hmer_clip,0)
        other.direction = 'reference'
        return other

    def _key2str(self):
        return ''.join(np.repeat(self.flow_order, self.key))

    def haplotype_matching(self, read:FlowBasedRead) -> np.float:
        '''Returns log likelihood for the match of the read and the haplotype. 
        Both the haplotype and the read flow matrices should be aligned in the reference direction.
        It is assumed that haplotype does not have probabilities.

        Parameters
        ----------
            read: FlowBasedRead
                flow based read **after `apply_alignment`**. 
        Returns
        -------
        np.float
            Log likelihood of the match
        '''
        assert self.direction == read.direction \
                and self.direction == "reference", \
                    "Only reads aligned to the reference please"

        assert self.start <= read.start and self.end >= read.end, \
                    "Read alignment should be contained in the haplotype"

        hap_locs = np.array([x[0] for x in self.record.get_aligned_pairs(matches_only=True)])
        ref_locs = np.array([x[1] for x in self.record.get_aligned_pairs(matches_only=True)])
        hap_start_loc = hap_locs[np.searchsorted(ref_locs, read.start)]
        hap_end_loc = hap_locs[np.searchsorted(ref_locs, read.end-1)]

        left_clip = hap_start_loc
        right_clip = len(self.seq) - hap_end_loc - 1
        right_clip = max(0,right_clip)

        clipping = [] 
        if left_clip > 0 :
            clipping.append((5,left_clip))
        clipping.append((0, len(read.seq)))
        if right_clip > 0 : 
            clipping.append((5,right_clip))

        if left_clip < 0 or right_clip < 0 or left_clip >= len(self.seq) or right_clip >= len(self.seq):
            return 1

        if self.key.max() >= 9 : 
            return -np.Inf

        clip_left, left_hmer_clip = self._left_clipped_flows(clipping)
        clip_right, right_hmer_clip = self._right_clipped_flows(clipping)
        
        assert abs(left_hmer_clip) < 11 and abs(right_hmer_clip) < 11 , "Weird hmer_clip"
        if clip_left >= len(self.key) or clip_right >= len(self.key): 
            return -np.Inf

        assert left_hmer_clip >= 0 and right_hmer_clip >= 0, "Some problem with hmer clips"
        key = self.key.copy()
        original_length = len(key)
        clip_left = max(clip_left-4, 0)
        clip_right = min(original_length, original_length - clip_right+4)
        key = key[clip_left:clip_right]
        flow_order = self.flow_order[clip_left:clip_right]
        starting_points = np.nonzero(flow_order==read.flow_order[0])[0]
        starting_points = starting_points[starting_points+len(read.key) <= len(key)]
        best_alignment = -np.Inf
        for s in starting_points :
            fetch = np.log10(read._flow_matrix[np.clip(key[s:s+len(read.key)], None, 
                        read._flow_matrix.shape[0]-1),np.arange(len(read.key))])[1:-1].sum()
            if fetch > best_alignment:
                best_alignment = fetch
        return best_alignment

def get_haplotype_by_read_matrix( haplotypes: list, reads: list ) -> np.ndarray : 
    '''Matrix of likelihoods of reads from each haplotype

    Parameters
    ----------
    haplotypes: list
        List of haplotypes (n_hap FlowBasedReads)
    reads: list
        List of reads (n_reads FlowBasedReads)

    Returns
    -------
    n_hapxn_reads matrix of likelihood log(P(read|hap)
    '''

    n_hap = len(haplotypes)
    n_reads = len(reads)
    result = np.zeros((n_hap, n_reads))
    for i in range(n_hap):
        for j in range(n_reads): 
            result[i,j] = haplotypes[i].haplotype_matching(reads[j])
    return result






