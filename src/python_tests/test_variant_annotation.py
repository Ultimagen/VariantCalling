import subprocess
from os.path import join as pjoin

import numpy as np
import pandas as pd
import pyBigWig as pbw
import pyfaidx
import pysam

import python.modules.variant_annotation as variant_annotation
from pathmagic import PYTHON_TESTS_PATH, COMMON


class TestVariantAnnotation:
    def test_get_coverage(self, tmpdir):
        temp_bw_name1 = self._create_temp_bw(tmpdir, "test1.bw", 20)
        temp_bw_name2 = self._create_temp_bw(tmpdir, "test2.bw", 0)
        df = self._create_test_df_for_coverage()

        result = variant_annotation.get_coverage(
            df.copy(), [temp_bw_name1], [temp_bw_name2])
        expected_total, expected_well_mapped = self._create_expected_coverage()
        assert result.shape == (df.shape[0], df.shape[1] + 3)
        assert 'coverage' in result.columns
        assert 'well_mapped_coverage' in result.columns
        assert 'repetitive_read_coverage' in result.columns
        assert np.all(result['coverage'] - result['well_mapped_coverage']
                      == result['repetitive_read_coverage'])
        # we calculate coverage on the same BAM twice, so the coverage should be twice the expected
        assert np.all(result['coverage'] == expected_total)
        assert np.all(result['well_mapped_coverage'] == expected_well_mapped)

    def test_get_coverage_empty_dataframe(self, tmpdir):
        temp_bw_name1 = self._create_temp_bw(tmpdir, "test1.bw", 20)
        temp_bw_name2 = self._create_temp_bw(tmpdir, "test2.bw", 0)
        df = self._create_test_df_for_coverage().iloc[:0, :]

        result = variant_annotation.get_coverage(
            df.copy(), [temp_bw_name1], [temp_bw_name2])
        assert result.shape == (df.shape[0], df.shape[1] + 3)

    # Temporary bam contains read that starts on each location, every second read is duplicate (should be discarded)
    # Every third read is of low mapping quality. _create_expected_coverage generates the expected coverage profile
    # now instead of BAM we already collect the coverage in the BW
    def _create_temp_bam(self, tmpdir, name):
        header = {'HD': {'VN': '1.0'},
                  'SQ': [{'LN': 100000, 'SN': 'chr20'}]}
        fai = pyfaidx.Fasta(pjoin(PYTHON_TESTS_PATH, COMMON, "sample.fasta"))
        with pysam.AlignmentFile(pjoin(tmpdir, name), "wb", header=header) as outf:
            for i in range(90000, 91000):
                a = pysam.AlignedSegment()
                a.query_name = f"read_{i}"
                a.query_sequence = str(fai['chr20'][i])
                a.query_qualities = pysam.qualitystring_to_array("I")
                a.is_duplicate = i % 2
                a.reference_id = 0
                a.reference_start = i
                a.cigar = ((0, 1),)
                a.mapping_quality = 30 * (i % 3)
                outf.write(a)
        subprocess.check_call(['samtools', 'index', pjoin(tmpdir, name)])
        return pjoin(tmpdir, name)

    def _create_temp_bw(self, tmpdir, name, quality_threshold):
        with pbw.open(pjoin(tmpdir, name), 'w') as outf:
            outf.addHeader([('chr20', 100000)])
            for loc in range(90000, 91000):
                outf.addEntries(["chr20"], [loc], ends=[loc+1],
                                values=[float(((loc+1) % 2) * ((30*(loc % 3)) >= quality_threshold))])
            outf.addEntries(["chr20"], [91000], ends=[92000], values=[0.0])
        return pjoin(tmpdir, name)

    # creates dataframe that would test coverage
    def _create_test_df_for_coverage(self):
        df = pd.DataFrame(
            {'chrom': ['chr20'] * 2000, 'pos': np.arange(90000, 92000)+1})  # note  that VCF is one-based
        df.set_index(
            df.apply(lambda x: (x['chrom'], x['pos']), axis=1), inplace=True)
        return df

    # This creates the expected coverage from the bam simulated by _create_temp_bam
    def _create_expected_coverage(self):
        result = np.ones(1000, dtype=np.int)
        result[((np.arange(90000, 91000)) % 2) == 1] = 0
        result_well_mapped = result.copy()
        result_well_mapped[((np.arange(90000, 91000)) % 3) == 0] = 0
        result = np.concatenate((result, np.zeros(1000)))
        result_well_mapped = np.concatenate(
            (result_well_mapped, np.zeros(1000)))
        return result, result_well_mapped

    def test_cycleskip_status(self):
        df, result = self._create_cycleskip_test_dataframe()
        df = variant_annotation.annotate_cycle_skip(df, "TGCA")
        assert np.all(np.array(df['cycleskip_status']) == np.array(result))

    def _create_cycleskip_test_dataframe(self):
        indel_status = [True, False, False, False, False, False]
        alleles = [('CA',), ('G', 'C'), ('C',), ('C',), ('G',), ('G',)]
        ref = ['C', 'T', 'A', 'A', 'A', 'C']
        alleles = [(ref[i],)+alleles[i] for i in range(len(ref))]
        left_motif = ['TGGCA', 'TGGCA', 'TGGCA', 'TGGCA', 'TGGCT', 'TGGCT']
        right_motif = ['TGGCA', 'TGGCA', 'AGCTA', 'CGCTA', 'CGGCA', 'AGGCA']
        cycleskip_status = [variant_annotation.UNDETERMINED,
                            variant_annotation.UNDETERMINED,
                            'cycle-skip', 'non-skip', 'cycle-skip',
                            'possible-cycle-skip']
        df = pd.DataFrame({'indel': indel_status,
                           'alleles': alleles, 'ref': ref, 'left_motif': left_motif, 'right_motif': right_motif})
        result = pd.Series(cycleskip_status)
        return df, result
