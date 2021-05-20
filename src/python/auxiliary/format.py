from pandas.api.types import CategoricalDtype

CHROM_DTYPE = CategoricalDtype(
    categories=[f"chr{j}" for j in range(1, 23)] + ["chrX", "chrY", "chrM"], ordered=True
)
