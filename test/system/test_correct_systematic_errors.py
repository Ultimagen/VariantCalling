import os
from os.path import dirname
from test import get_resource_dir

import pysam

from ugvc.pipelines.sec import correct_systematic_errors
from ugvc.sec.systematic_error_correction_call import SECCallType


class TestCorrectSystematicErrors:
    inputs_dir = get_resource_dir(__file__)

    def test_correct_systematic_errors_glob(self, tmpdir):
        output_file = f"{tmpdir}/HG00239.vcf.gz"
        os.makedirs(dirname(output_file), exist_ok=True)

        correct_systematic_errors.run(
            [
                "correct_systematic_errors",
                "--relevant_coords",
                f"{self.inputs_dir}/blacklist_hg001_10s.bed",
                "--model",
                f"{self.inputs_dir}/conditional_allele_distribution.*.pkl",
                "--gvcf",
                f"{self.inputs_dir}/HG00239.g.vcf.gz",
                "--output_file",
                output_file,
            ]
        )
        vcf = pysam.VariantFile(output_file)
        sec_types = {}
        known_variants = set()
        novel_variants = set()
        unobserved_noise_sites = set()
        for record in vcf:
            sec_type = record.samples[0]["ST"]
            if sec_type in sec_types:
                sec_types[sec_type] += 1
            else:
                sec_types[sec_type] = 1
            if sec_type == SECCallType.KNOWN.value:
                known_variants.add(record.pos)
            elif sec_type == SECCallType.NOVEL.value:
                novel_variants.add(record.pos)
            elif sec_type == SECCallType.UNOBSERVED.value:
                unobserved_noise_sites.add(record.pos)

        assert {
            "known": 7,
            "novel": 4,
            "reference": 19336,
            "uncorrelated": 130,
            "unobserved": 43,
        } == sec_types

        positives = {
            54781424,
            177932523,
            70741007,
            84237494,
            299411,
            56700405,
            56701244,
            52434034,
            93045059,
            50304411,
            16353372,
            21415639,
            41140109,
            54574304,
            54574902,
            54574903,
            54574907,
            54574968,
            54574979,
            54574980,
            54574982,
            55481893,
            227027903,
            41878845,
            21636783,
            97475398,
            112534211,
            24808471,
            100951865,
            100951872,
            100951876,
            100951887,
            100951890,
            100951897,
            100951900,
            100951903,
            100951907,
            100951908,
            100951933,
            100951943,
            100951952,
            100951953,
            100965843,
            100965859,
            17001889,
            29927416,
            21415757,
        }

        # TP
        assert 4 == len(known_variants.intersection(positives))
        assert 41 == len(unobserved_noise_sites.intersection(positives))
        assert 0 == len(novel_variants.intersection(positives))

        # FP
        assert {119471049, 119471038, 119471054} == known_variants.difference(positives)
        assert {41332392, 158829546} == unobserved_noise_sites.difference(positives)
        assert {35995512, 131838289, 144504645, 2372430} == novel_variants.difference(positives)

        # FN
        assert 2 == len(
            positives.difference(
                positives.intersection(novel_variants.union(known_variants).union(unobserved_noise_sites))
            )
        )

    def test_correct_systematic_errors_single_chr(self, tmpdir):
        output_file = f"{tmpdir}/HG00239.chr3.vcf.gz"

        correct_systematic_errors.run(
            [
                "correct_systematic_errors",
                "--relevant_coords",
                f"{self.inputs_dir}/blacklist_hg001_10s.chr3.bed",
                "--model",
                f"{self.inputs_dir}/conditional_allele_distribution.chr3.pkl",
                "--gvcf",
                f"{self.inputs_dir}/HG00239.g.vcf.gz",
                "--output_file",
                output_file,
            ]
        )
        vcf = pysam.VariantFile(output_file)
        sec_types = {}
        known_variants = set()
        novel_variants = set()
        unobserved_noise_sites = set()
        for record in vcf:
            sec_type = record.samples[0]["ST"]
            if sec_type in sec_types:
                sec_types[sec_type] += 1
            else:
                sec_types[sec_type] = 1
            if sec_type == SECCallType.KNOWN.value:
                known_variants.add(record.pos)
            elif sec_type == SECCallType.NOVEL.value:
                novel_variants.add(record.pos)
            elif sec_type == SECCallType.UNOBSERVED.value:
                unobserved_noise_sites.add(record.pos)

        assert {"reference": 936, "unobserved": 2} == sec_types
