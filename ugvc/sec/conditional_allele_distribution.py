from __future__ import annotations

from ugbio_core.strand_direction import StrandDirection
from ugvc.sec.read_counts import ReadCounts


class ConditionalAlleleDistribution:
    """
    Represent distributions of read counts for a specific locus (locus is not saved in object to save memory)
    For each ground-truth genotype,
       For each observed alleles list:
           how many reads were observed in each strand for each allele?
           how many samples showed these observed_alleles?
    Object is constructed for a single sample, and updated with other samples as needed

    An internal key is calculated for observed alleles such that the counts are invariant to the order of alt alleles
    """

    def __init__(
        self,
        conditioned_alleles: str,
        conditioned_genotype: str,
        observed_alleles: str,
        allele_counts: dict[str, ReadCounts],
        num_of_samples_with_observed_alleles: int = 1,
    ):
        self.conditioned_alleles = conditioned_alleles
        self.allele_counts_dict = {conditioned_genotype: {}}
        self.num_of_samples_with_alleles = {conditioned_genotype: {}}
        observed_alleles_key = self.define_observed_alleles_key(observed_alleles)
        self.allele_counts_dict[conditioned_genotype] = {observed_alleles_key: allele_counts}
        self.num_of_samples_with_alleles[conditioned_genotype] = {
            observed_alleles_key: num_of_samples_with_observed_alleles
        }

    def update_distribution(self, other) -> None:
        # pylint: disable=too-many-nested-blocks
        for (
            conditioned_genotype,
            observed_allele_counts,
        ) in other.allele_counts_dict.items():
            if conditioned_genotype not in self.num_of_samples_with_alleles:
                self.num_of_samples_with_alleles[conditioned_genotype] = other.num_of_samples_with_alleles[
                    conditioned_genotype
                ]
                self.allele_counts_dict[conditioned_genotype] = other.allele_counts_dict[conditioned_genotype]
            else:
                for observed_alleles, allele_counts in observed_allele_counts.items():
                    if observed_alleles in self.num_of_samples_with_alleles[conditioned_genotype]:
                        self_allele_counts = self.allele_counts_dict[conditioned_genotype][observed_alleles]
                        for allele, read_counts in allele_counts.items():
                            if allele in self_allele_counts:
                                self_allele_counts[allele].add_counts(read_counts)
                            else:
                                self_allele_counts[allele] = read_counts
                        self.num_of_samples_with_alleles[conditioned_genotype][observed_alleles] += 1
                    else:
                        self.allele_counts_dict[conditioned_genotype][observed_alleles] = allele_counts
                        self.num_of_samples_with_alleles[conditioned_genotype][observed_alleles] = 1

    def get_allele_count(
        self,
        conditioned_genotype: str,
        observed_alleles: str,
        allele: str,
        strand: StrandDirection,
    ):
        observed_alleles_key = self.define_observed_alleles_key(observed_alleles)
        return self.allele_counts_dict[conditioned_genotype][observed_alleles_key][allele].get_count(strand)

    def get_allele_counts(self, conditioned_genotype: str, observed_alleles: str, allele: str) -> ReadCounts:
        observed_alleles_key = self.define_observed_alleles_key(observed_alleles)
        return self.allele_counts_dict[conditioned_genotype][observed_alleles_key][allele]

    def get_allele_counts_string(self, conditioned_genotype: str, observed_alleles: str):
        observed_alleles_key = self.define_observed_alleles_key(observed_alleles)
        allele_counts_fields = []
        for allele, read_counts in self.allele_counts_dict[conditioned_genotype][observed_alleles_key].items():
            allele_counts_fields.extend([allele, read_counts.get_counts_tuple_as_str()])
        return " ".join(allele_counts_fields)

    def get_observed_alleles_frequency(self, conditioned_genotype: str, observed_alleles: str) -> float:
        observed_alleles_key = self.define_observed_alleles_key(observed_alleles)
        total_count = sum(self.num_of_samples_with_alleles[conditioned_genotype].values())
        if (
            observed_alleles_key in self.num_of_samples_with_alleles[conditioned_genotype]
            and self.num_of_samples_with_alleles[conditioned_genotype][observed_alleles_key] > 0
        ):
            return self.num_of_samples_with_alleles[conditioned_genotype][observed_alleles_key] / total_count
        return min(1 / (total_count + 1), 0.01)

    def get_possible_observed_alleles(self, conditioned_genotype: str) -> set[str]:
        return set(self.num_of_samples_with_alleles[conditioned_genotype].keys())

    def get_observed_alleles_counts_list(self, conditioned_genotype: str, observed_alleles: str) -> list[int]:
        """
        For a specific set of conditioned_genotype + observed_alleles (represented strings),
        return a list of [+count_0, -count_0, ?count_0 ...]
        in the order of given observed_alleles comma-separated string
        """
        observed_alleles_key = self.define_observed_alleles_key(observed_alleles)
        actual_allele_counts = self.allele_counts_dict[conditioned_genotype][observed_alleles_key]
        return get_allele_counts_list(actual_allele_counts, observed_alleles_key)

    def get_string_records(self, chrom: str, pos: int):
        records = []
        # pylint: disable=consider-using-dict-items
        for conditioned_genotype in self.num_of_samples_with_alleles:
            for observed_alleles, count in self.num_of_samples_with_alleles[conditioned_genotype].items():
                counts_str = self.get_allele_counts_string(conditioned_genotype, observed_alleles)
                records.append(
                    f"{chrom}\t{pos}\t{self.conditioned_alleles}\t{conditioned_genotype}"
                    f"\t{observed_alleles}\t{count}\t{counts_str}"
                )
        return records

    def __str__(self):
        return f"{self.conditioned_alleles} {self.num_of_samples_with_alleles} {self.allele_counts_dict}"

    @staticmethod
    def define_observed_alleles_key(observed_alleles: str):
        """
        generate allele order-invariant key
        """
        if observed_alleles.count(",") <= 1:
            return observed_alleles
        # sort alts in multi-allelic positions
        observed_alleles_list = observed_alleles.split(",")
        sorted_alts = ",".join(sorted(observed_alleles_list[1:]))
        return f"{observed_alleles_list[0]},{sorted_alts}"


def get_allele_counts_list(actual_allele_counts: dict[str, ReadCounts], observed_alleles: str):
    """
    return a list of [allele_0, +count_0, -count_0, ...]
    in the order of given observed_alleles comma-separated string
    """
    actual_distribution_list = []
    for allele in observed_alleles.split(","):
        if allele in actual_allele_counts:
            actual_distribution_list.append(actual_allele_counts[allele].get_count(StrandDirection.FORWARD))
            actual_distribution_list.append(actual_allele_counts[allele].get_count(StrandDirection.REVERSE))
        else:
            actual_distribution_list.extend([0, 0])
    return actual_distribution_list
