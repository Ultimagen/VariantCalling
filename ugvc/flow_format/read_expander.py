# Deprecated
from __future__ import annotations

import itertools

import pysam

# Various functions of adding variants to reads


def parse_alt(alt: str) -> list:
    variants = alt.split(";")
    variants_parsed = [x.split(",") for x in variants]
    variants_tup = [(int(x[0]), x[1], x[2], float(x[3])) for x in variants_parsed]
    return variants_tup


def validate_combination(mutation_comb: tuple):
    poss = [x[0] for x in mutation_comb]
    if len(poss) == len(set(poss)):
        return True
    return False


def get_read_variant_str(reference, mutations):
    mutated = ""
    cur_source_pos = 0
    cur_dest_pos = 0
    if len(mutations) == 0:
        return reference
    expected_diff = sum([len(x[2]) - len(x[1]) for x in mutations], 0)
    expected_dest_length = len(reference) + expected_diff
    for mutation in mutations:
        update = mutation[0] - cur_source_pos
        if update >= 0:
            mutated += reference[cur_source_pos : cur_source_pos + update]
            cur_source_pos += update
            cur_dest_pos += update
        else:
            cur_source_pos = cur_source_pos + update
            cur_dest_pos = cur_dest_pos + update
            if cur_dest_pos < len(mutated):
                mutated = mutated[:cur_dest_pos]

        mutated += mutation[2]
        assert (
            reference[cur_source_pos : cur_source_pos + len(mutation[1])] == mutation[1]
        ), "Reference different from \
                         the ref allele"
        cur_source_pos += len(mutation[1])
        cur_dest_pos += len(mutation[2])

    mutated += reference[cur_source_pos:]
    assert (
        len(mutated) == expected_dest_length
    ), "Resultant sequence \
    different in length from expected"

    return mutated


class ReadExpander:
    def __init__(self, read: pysam.AlignedSegment):
        self.read = read.query_sequence
        self.alts = read.get_tag("AL")
        if not self.alts:
            self.alts = []
        else:
            self.alts = parse_alt(self.alts)

    def get_all_read_expansions(self):
        all_variant_combinations = sum(
            [list(itertools.combinations(self.alts, n)) for n in range(0, len(self.alts) + 1)],
            [],
        )
        all_variant_combinations = [x for x in all_variant_combinations if validate_combination(x)]
        read_variants = [self.get_read_variant(x) for x in all_variant_combinations]
        return read_variants

    def get_read_variant(self, mutations):
        reference = self.read
        return get_read_variant_str(reference, mutations)
