# Set of classes for working with pileup in a flow space.
# Allows to fetch the flow probabilities for each hompolymer in the pileup.
import pysam
import pysam.Pileup

import ugvc.flow_format.flow_based_read as fbr


class FlowBasedIteratorColumn:
    """Wrapper for pysam.IteratorColumn that allows to fetch flow probabilities for each homopolymer in the pileup.

    Attributes
    ----------
    pileupIterator : pysam.IteratorColumn
        Original pysam.IteratorColumn object
    """

    def __init__(self, pileupIterator: pysam.IteratorColumn):
        self.pileupIterator = pileupIterator
        self.__flow_reads_dict = {}

    def __next__(self):
        result = next(self.pileupIterator)
        pileups = result.pileups

        if len(self.__flow_reads_dict) > 1000:
            qnames = [x.alignment.query_name for x in pileups]
            delete_candidates = []
            for n in self.__flow_reads_dict:
                if n not in qnames:
                    delete_candidates.append(n)
            for n in delete_candidates:
                del self.__flow_reads_dict[n]
        for x in pileups:
            if x.alignment.query_name not in self.__flow_reads_dict:
                fr = fbr.FlowBasedRead.from_sam_record(x.alignment, max_hmer_size=20)
                self.__flow_reads_dict[x.alignment.query_name] = fr
        return FlowBasedPileupColumn(result, self.__flow_reads_dict)


class FlowBasedPileupColumn:
    """Wrapper for pysam.PileupColumn that allows to fetch flow probabilities for each homopolymer in the pileup.

    Attributes
    ----------
    pc : pysam.PileupColumn
        Original pysam.PileupColumn object
    flow_reads_dict : dict
        Dictionary of FlowBasedRead objects, indexed by query name

    Methods
    -------
    fetch_hmers():
        Returns a list of flow probabilities for each homopolymer in the pileup.
    """

    def __init__(self, pc: pysam.PileupColumn, flow_reads: dict):
        """Constructor - receives a pysam.PileupColumn object and a dictionary of FlowBasedRead objects.
        The dictionary is query_name: FlowBasedRead

        Parameters
        ----------
        pc : pysam.PileupColumn
            PileupColumn
        flow_reads : dict
            pysam.AlignedRead reads converted into FlowBasedRead objects
        """

        self.pc = pc
        self.flow_reads_dict = flow_reads

    def fetch_hmer_qualities(self) -> list[tuple]:
        """
        Return list of hmer length probabilities for every read in the PileupColumn

        Return
        ------
        list[tuple]:
            list of pairs (hmer,probabilities of hmer length) for every read  in the PileupColumn

        See also
        --------
        flow_based_read.FlowBasedRead.get_flow_matrix_column_for_base
        """
        qpos = [x.query_position_or_next for x in self.pc.pileups]
        qnames = [x.alignment.query_name for x in self.pc.pileups]
        hmers = [self.flow_reads_dict[x].get_flow_matrix_column_for_base(y) for x, y in zip(qnames, qpos)]
        return hmers


class FlowBasedAlignmentFile(pysam.AlignmentFile):
    """Wrapper for pysam.AlignmentFile that returns FlowBasedPileupColumn objects.

    Methods
    -------
    pileup(contig, start, end, mq):
        similar to pysam.AlignmentFile.pileup, but returns FlowBasedPileupColumn objects.
        Works only in `truncate` mode and min_base_quality = 0

    See also
    --------
    pysam.AlignmentFile
    """

    def pileup(self, contig, start, end, mq) -> FlowBasedIteratorColumn:
        """Return a generator of FlowBasedPileupColumn objects.
        Parameters
        ----------
        contig : str
            Reference sequence name
        start : int
            Start position (1-based)
        end : int
            End position (1-based)
        mq : int
            Minimum mapping quality

        Returns
        -------
        FlowBasedIteratorColumn
            Iterator of FlowBasedPileupColumn objects
        """
        pup = super().pileup(
            contig, start, end, truncate=True, min_base_quality=0, flag_filter=3844, min_mapping_quality=mq
        )
        return FlowBasedIteratorColumn(pup)
