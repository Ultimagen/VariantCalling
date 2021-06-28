from pandas.api.types import CategoricalDtype

CHROM_DTYPE = CategoricalDtype(
    categories=[f"chr{j}" for j in range(1, 23)] + ["chrX", "chrY", "chrM"], ordered=True
)

CYCLE_SKIP = "cycle-skip"
POSSIBLE_CYCLE_SKIP = "possible-cycle-skip"
NON_CYCLE_SKIP = "non-skip"

CYCLE_SKIP_DTYPE = CategoricalDtype(
    categories=[CYCLE_SKIP, POSSIBLE_CYCLE_SKIP, NON_CYCLE_SKIP], ordered=True
)