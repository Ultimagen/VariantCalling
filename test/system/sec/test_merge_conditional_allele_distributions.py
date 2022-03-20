import glob
import os
import unittest
from os.path import dirname

from ugvc.sec.conditional_allele_distributions import ConditionalAlleleDistributions
from test import test_dir, get_resource_dir, make_test_outputs_dir
from ugvc.pipelines.sec.merge_conditional_allele_distributions import merge_conditional_allele_distributions


class TestMergeConditionalAlleleDistributions(unittest.TestCase):
    inputs_dir = get_resource_dir(__name__)
    test_outputs_dir = make_test_outputs_dir(__name__)
    os.makedirs(test_outputs_dir, exist_ok=True)

    def test_merge_conditional_allele_distributions(self):

        output_file = f'{test_dir}/test_outputs/exome_hg001_10s/correction/HG00239.vcf.gz'
        os.makedirs(dirname(output_file), exist_ok=True)

        test_outputs_dir = f'{test_dir}/test_outputs/sec/test_merge_conditional_allele_distribution'
        os.makedirs(test_outputs_dir, exist_ok=True)
        cad_files_path = f'{test_outputs_dir}/conditional_distribution_files.txt'
        output_prefix = f'{test_outputs_dir}/conditional_allele_distributions'
        with open(cad_files_path, 'w') as cad_files_fh:
            cad_files_fh.write(f'{self.inputs_dir}/HG00096.head.tsv\n')
            cad_files_fh.write(f'{self.inputs_dir}/HG00140.head.tsv\n')

        merge_conditional_allele_distributions(['--conditional_allele_distribution_files', cad_files_path,
                                                '--output_prefix', output_prefix])
        pickle_files = glob.glob(f'{output_prefix}.*.pkl')
        cads = ConditionalAlleleDistributions(pickle_files)

        cads_list_chr1 = [cad for cad in cads.distributions_per_chromosome['chr1'].values()]
        cads_list_chr2 = [cad for cad in cads.distributions_per_chromosome['chr2'].values()]
        cads_list_chr3 = [cad for cad in cads.distributions_per_chromosome['chr3'].values()]

        # 87 common records (chr, pos)
        self.assertEqual(119, len(cads_list_chr1 + cads_list_chr2 + cads_list_chr3))

        # few records from chr2, and chr3 were read
        self.assertEqual(7, len(cads_list_chr2))
        self.assertEqual(6, len(cads_list_chr3))

        cad = cads.get_distributions_per_locus('chr1', 930192)
        self.assertEqual({'T,TG'}, cad.get_possible_observed_alleles('0/0'))
        self.assertEqual((13, 55, 0), cad.get_allele_counts('0/0', 'T,TG', 'T').get_counts())
        self.assertEqual((42, 0, 0), cad.get_allele_counts('0/0', 'T,TG', 'TG').get_counts())

        cad = cads.get_distributions_per_locus('chr1', 942741)
        self.assertEqual({'CGG,C', 'CG,C'}, cad.get_possible_observed_alleles('0/0'))
        self.assertEqual((5, 3, 0), cad.get_allele_counts('0/0', 'CGG,C', 'CGG').get_counts())
        self.assertEqual((5, 0, 0), cad.get_allele_counts('0/0', 'CGG,C', 'C').get_counts())
        self.assertEqual((1, 7, 0), cad.get_allele_counts('0/0', 'CG,C', 'CG').get_counts())
        self.assertEqual((1, 0, 0), cad.get_allele_counts('0/0', 'CG,C', 'C').get_counts())
