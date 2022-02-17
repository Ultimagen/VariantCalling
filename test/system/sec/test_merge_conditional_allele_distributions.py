import os
import pickle
import unittest
from os.path import dirname

from sec.conditional_allele_distributions import ConditionalAlleleDistributions
from test import test_dir
from ugvc.pipelines.sec.merge_conditional_allele_distributions import merge_conditional_allele_distributions


class TestMergeConditionalAlleleDistributions(unittest.TestCase):

    def test_merge_conditional_allele_distributions(self):
        proj_dir = f'{test_dir}/resources/sec'
        output_file = f'{test_dir}/test_outputs/exome_hg001_10s/correction/HG00239.vcf.gz'
        os.makedirs(dirname(output_file), exist_ok=True)

        test_outputs_dir = f'{test_dir}/test_outputs/sec/test_merge_conditional_allele_distribution'
        os.makedirs(test_outputs_dir, exist_ok=True)
        cad_files_path = f'{test_outputs_dir}/conditional_distribution_files.txt'
        output_file = f'{test_outputs_dir}/conditional_allele_distributions.pkl'
        with open(cad_files_path, 'w') as cad_files_fh:
            cad_files_fh.write(f'{proj_dir}/HG00096.head.tsv\n')
            cad_files_fh.write(f'{proj_dir}/HG00140.head.tsv\n')

        merge_conditional_allele_distributions(['--conditional_allele_distribution_files', cad_files_path,
                                                '--output_file', output_file])

        with open(output_file, 'rb') as db:
            cads: ConditionalAlleleDistributions = pickle.load(db)

        cads_list = [cad for cad in cads.distributions_per_chromosome['chr1'].values()]

        # 87 common records (chr, pos, gt_alleles, called_alleles)
        self.assertEqual(113, len(cads_list))

        cad = cads.get_distributions_per_locus('chr1', 930192)
        self.assertEqual({'T,TG'}, cad.get_possible_observed_alleles('0/0'))
        self.assertEqual((19, 85, 0), cad.get_allele_counts('0/0', 'T,TG', 'T').get_counts())
        self.assertEqual((60, 0, 0), cad.get_allele_counts('0/0', 'T,TG', 'TG').get_counts())

        cad = cads.get_distributions_per_locus('chr1', 942741)
        self.assertEqual({'CGG,C', 'CG,C'}, cad.get_possible_observed_alleles('0/0'))
        self.assertEqual((10, 6, 0), cad.get_allele_counts('0/0', 'CGG,C', 'CGG').get_counts())
        self.assertEqual((10, 0, 0), cad.get_allele_counts('0/0', 'CGG,C', 'C').get_counts())
        self.assertEqual((1, 7, 0), cad.get_allele_counts('0/0', 'CG,C', 'CG').get_counts())
        self.assertEqual((1, 0, 0), cad.get_allele_counts('0/0', 'CG,C', 'C').get_counts())
