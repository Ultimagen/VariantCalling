import pytest

from test import get_resource_dir, test_dir
from ugvc.pipelines import vcfeval_flavors


class TestVcfEvalFlavors:
    inputs_dir = get_resource_dir(__file__)
    general_inputs_dir = f'{test_dir}/resources/general/chr1_head'

    @pytest.mark.parametrize('penalty,expected_tp, expected_fp, expected_fn, expected_precision, expected_recall',
                             [
                                 [2, 24, 6, 7, 80.0, 77.42],
                                 [1, 24, 5.5, 6.5, 81.36, 78.69],
                                 [0, 24, 5, 6, 82.76, 80.0],
                                 [-1, 25, 5, 6, 83.33, 80.65],
                             ])
    def test_penalty(self, tmpdir,
                     penalty,
                     expected_tp,
                     expected_fp,
                     expected_fn,
                     expected_precision,
                     expected_recall):
        """
        We've implemented two indel allele error:
          (chr1    805514  .       AC      A)
          (chr1	842922	rs58115377	C	CCTT)
        And one indel genotype error:
          (chr1    814583  rs56197012      T       TAA )
        These should affect the evaluation depending on the penalty score
        """
        out_dir = f'{tmpdir}/test_penalty_{penalty}'
        result = vcfeval_flavors.run(
            [
                'vcfeval_flavors',
                '-c', f'{self.inputs_dir}/004777-UGAv3-20.pred.chr1_1-1000000.vcf.gz',
                '-b', f'{self.inputs_dir}/HG001_GRCh38_1_22_v4.2.1_benchmark.chr1_1-1000000.vcf.gz',
                '-e', f'{self.inputs_dir}/HG001_GRCh38_1_22_v4.2.1_benchmark.chr1_1-1000000.bed',
                '-o', out_dir,
                '-t', f'{self.general_inputs_dir}/Homo_sapiens_assembly38.fasta.sdf',
                '-p', str(penalty)
            ]
        )
        vtype, tp, fp, fn, precision, recall, f1 = result[1].split()
        assert vtype == 'indels'
        assert float(tp) == expected_tp
        assert float(fp) == expected_fp
        assert float(fn) == expected_fn
        assert float(precision) == expected_precision
        assert float(recall) == expected_recall
