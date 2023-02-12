import os
import tempfile

import pysam
import pytest

import ugvc.vcfbed.pysam_utils as pysam_utils


class TestPysamUtils:
    def _generate_indel_tests():
        vcfh = pysam.VariantHeader()
        vcfh.add_sample("ahstram")
        vcfh.add_meta("FILTER", items=[("ID", "RF"), ("Description", "Variant failed filter due to low RF")])
        vcfh.add_meta("contig", items=[("ID", 1)])
        vcfh.add_meta("FORMAT", items=[("ID", "GT"), ("Number", 1), ("Type", "String"), ("Description", "Genotype")])
        tmpfilename = tempfile.mktemp(suffix="vcf")
        vcf = pysam.VariantFile(tmpfilename, "w", header=vcfh)

        records = []
        r = vcf.new_record(contig=str(1), start=999, stop=1000, alleles=("A", "T"), filter="RF")
        records.append(r)
        r = vcf.new_record(contig=str(1), start=999, stop=1000, alleles=("A", "AT"), filter="RF")
        records.append(r)
        r = vcf.new_record(contig=str(1), start=999, stop=1000, alleles=("AT", "A"), filter="RF")
        records.append(r)
        r = vcf.new_record(contig=str(1), start=999, stop=1000, alleles=("AT", "A", "AG", "ATC"), filter="RF")
        records.append(r)
        r = vcf.new_record(contig=str(1), start=999, stop=1000, alleles=("AT", "A", "<NON_REF>"), filter="RF")
        records.append(r)

        os.unlink(tmpfilename)
        return records

    test_inputs = _generate_indel_tests()

    test_alleles = ("AT", "A", "<NON_REF>")

    @pytest.mark.parametrize("input,expected", zip(test_alleles, [False, False, True]))
    def test_is_symbolic(self, input, expected):
        assert pysam_utils.is_symbolic(input) == expected

    @pytest.mark.parametrize(
        "input,expected",
        zip(
            test_inputs,
            [[False, False], [False, True], [False, True], [False, True, False, True], [False, True, False]],
        ),
    )
    def test_is_indel(self, input, expected):
        assert pysam_utils.is_indel(input) == expected

    @pytest.mark.parametrize(
        "input,expected",
        zip(
            test_inputs,
            [[False, False], [False, False], [False, True], [False, True, False, False], [False, True, False]],
        ),
    )
    def test_is_deletion(self, input, expected):
        assert pysam_utils.is_deletion(input) == expected

    @pytest.mark.parametrize(
        "input,expected",
        zip(
            test_inputs,
            [[False, False], [False, True], [False, False], [False, False, False, True], [False, False, False]],
        ),
    )
    def test_is_insertion(self, input, expected):
        assert pysam_utils.is_insertion(input) == expected

    @pytest.mark.parametrize("input,expected", zip(test_inputs, [[0, 0], [0, 1], [0, 1], [0, 1, 0, 1], [0, 1, 0]]))
    def test_indel_length(self, input, expected):
        assert pysam_utils.indel_length(input) == expected
