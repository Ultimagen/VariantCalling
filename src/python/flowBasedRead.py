# Flow-based read/haplotype class
# This will be a class that would hold a read in flow base
from . import utils
from . import simulator
import numpy as np
import pandas as pd
import BeadsData

DEFAULT_ERROR_MODEL_FN = "/home/ilya/proj/VariantCalling/work/190628/error_model.r2d.hd5"
#DEFAULT_ERROR_MODEL = pd.read_hdf(DEFAULT_ERROR_MODEL_FN, "error_model")


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
    r_seq: str
        Reverse complement read sequence
    key: np.ndarray
        sequence in flow base
    flow2base: np.ndarray
        position of the last output sequence base _before_ each flow for forward key
    rkey: np.ndarray
        reverse complement sequence in flow base
    flow2rbase: np.ndarray
        position of the last output sequence base _before_ each flow for reverse key
    flow_order: str
        sequence of flows
    _motif_size: int
        size of motif (left or right)
    Methods
    -------
    apply_cigar: 
        Returns a new read with cigar applied (takes care of hard clipping and soft clipping)
    '''

    def __init__(self, read_name: str, read: str, error_model: pd.DataFrame,
                 flow_order: str=simulator.DEFAULT_FLOW_ORDER, motif_size: int=5):
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
            Error model from motif, hmer to probability that data with +1,-1,0 of that hmer generated this motif
        motif_size: int
            Size of the motif
        '''
        self.read_name = read_name
        self.seq = read
        self.key = BeadsData.BeadsData.generateKeyFromSequence(self.seq)
        self.flow2base = self._key2base(self.key)
        self.r_seq = utils.revcomp(read)
        self.rkey = BeadsData.BeadsData.generateKeyFromSequence(self.r_seq)
        self.flow2rbase = self._key2base(self.rkey)
        self.flow_order = simulator.getFlow2Base(flow_order, len(self.key))
        self._error_model = error_model
        self._get_max_hmer_size()
        self._motif_size = motif_size


    def _get_max_hmer_size(self) -> None:
        '''Returns maximal possible data hmer size from self._error_model_dict

        Parameters
        ----------
        None

        Returns
        -------
        None
        modifies self._max_hmer
        '''
        self._max_hmer = self._error_model.index.get_level_values('hmer_number').max()

    def read2FlowMatrix(self) -> tuple:
        '''Gets the hmerxflow probability matrix 
        matrix[i,j] = P(read_hmer==read_hmer[j] | data_hmer[j]==i)

        Paramters
        ---------
        None

        Returns 
        -------
        tuple
            np.ndarray, np.ndarray of matrices that correspond to the forward and the reverse sequence
        '''
        if not hasattr(self, "_flowMatrix"):
            self._flowMatrix = self._getSingleFlowMatrix(self.key, self.flow2base)
        if not hasattr(self, "_flowrMatrix"):
            self._flowrMatrix = self._getSingleFlowMatrix(self.rkey, self.flow2rbase)
        return self._flowMatrix, self._flowrMatrix

    def _getSingleFlowMatrix(self, key: np.ndarray, flow2base: np.ndarray, seq: str) -> np.ndarray:
        '''Returns matrix flow matrix for a given flow key

        Parameters
        ----------
        key: np.ndarray 
            Hmer for each flow
        flow2base : np.ndarray 
            For each flow - what was the last base output **before** the flow
        seq: str
            Sequence of the read
        Returns
        -------
        hmerxflow probability matrix 
        '''

        motifs_left = []
        for i in range(key.shape[0]):
            left_base = flow2base[i]
            if left_base < 0 :
                motifs_left.append('')
            else: 
                motifs_left.append(seq[max(left_base - self._motif_size + 1, 0):left_base + 1])
                assert seq[left_base] != self.flow_order[i], "Something wrong with motifs"
        motifs_right = []
        for i in range(key.shape[0]-1):

            right_base = flow2base[i + 1]
            motifs_right.append(seq[right_base + 1:right_base + self._motif_size + 1])
            assert seq[right_base + 1] != self.flow_order[i], "Something wrong with motifs"
        motifs_right.append('')
        index = pd.MultiIndex.from_arrays([motifs_left, key,
                                           self.flow_order, motifs_right],
                                          names=['left', 'hmer_number', 'hmer_letter', 'right'])
        pos = pd.Series(data=np.arange(len(key)), index=index)
        key = pd.Series(data=key, index=index)

        #return motifs_left, motifs_right
        tmp = self._error_model.loc[index.sort_values(), ["P(-1)", "P(0)", "P(+1)"]]
        tmp['pos'] = pos
        tmp['key'] = key

        flow_matrix = np.zeros((self._max_hmer, len(key)))
        
        a1 = np.concatenate((np.array(tmp['key']-1),
                            np.array(tmp['key']), 
                            np.array(tmp['key']+1)))
        p1 = np.tile(np.array(tmp['pos']),3)
        v1 = np.concatenate((np.array(tmp['P(-1)']),
                            np.array(tmp['P(0)']), 
                            np.array(tmp['P(+1)'])))

        take = a1>=0
        flow_matrix[a1[take], p1[take]] = v1[take]

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

    