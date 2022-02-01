import argparse
import pickle
from typing import OrderedDict

from ugvc.sec.conditional_allele_distributions import ConditionalAlleleDistributions


class ClassifySeqErrors:

    def __init__(self, trio_ref_bases: str, ref_allele: str, alt_allele: str):
        self.trio_ref_bases = trio_ref_bases
        self.base_before = trio_ref_bases[0]
        self.base_at = trio_ref_bases[1]
        self.base_after = trio_ref_bases[2]
        self.ref_allele = ref_allele
        self.alt_allele = alt_allele

    def is_seq_error(self):
        if self.is_insertion():
            unique_insert = self.get_unique_chars(self.get_insert())
            # ATG -> ATTG, ATG -> ATGG
            if unique_insert == self.base_at or unique_insert == self.base_after:
                return True
            # ATG -> ATTGG
            if len(unique_insert) == 2 and self.base_at == unique_insert[0] and self.base_after == unique_insert[1]:
                return True
        elif self.is_deletion():
            unique_deletion = self.get_unique_chars(self.get_deletion())
            if unique_deletion == self.base_after:
                return True
        elif self.is_snp():
            if self.alt_allele == self.base_before or self.alt_allele == self.base_after:
                return True
        else:
            return False


    def is_insertion(self):
        return len(self.alt_allele) > len(self.ref_allele)

    def is_deletion(self):
        return len(self.alt_allele) < len(self.ref_allele)

    def is_snp(self):
        return len(self.ref_allele) == 1 and len(self.alt_allele) == 1

    def get_unique_chars(self, string: str):
        return ''.join(OrderedDict.fromkeys(string).keys())

    def get_insert(self):
        return self.alt_allele[len(self.ref_allele):]

    def get_deletion(self):
        return self.ref_allele[len(self.alt_allele):]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='path to pickle file containing conditional allele distributions per position of interest')
    parser.add_argument('--ref_genome', help='path to fasta_file of reference genome', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    with open(args.model, 'rb') as fh:
        conditional_allele_distributions: ConditionalAlleleDistributions = pickle.load(fh)

    for chrom, pos, cad in conditional_allele_distributions:
        # for observed_alleles, allele_counts in cad.
        pass









