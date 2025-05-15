import dill as pickle
import numpy as np
from ugbio_core.h5_utils import read_hdf

from ugvc.reports.report_utils import ErrorType


class ReportDataLoader:
    def __init__(self, concordance_file: str, reference_version: str, exome_column_name: str):
        self.concordance_file = concordance_file
        self.reference_version = reference_version
        self.columns = self.__columns_subset(exome_column_name)
        self.rename_dict = self.__get_rename_dict()

    def load_concordance_df(self):
        df = read_hdf(
            self.concordance_file, key="all", skip_keys=["concordance", "input_args"], columns_subset=self.columns
        )
        df.rename(columns=self.rename_dict, inplace=True)
        df["fp"] = (df["call"] == "FP") | (df["call"] == "FP_CA")
        df["fn"] = (df["base"] == "FN") | (df["base"] == "FN_CA")
        df["tp"] = df["call"] == "TP"
        np.seterr(invalid="ignore")
        if "vaf" not in df.columns:
            df["vaf"] = df[["ad", "dp"]].apply(
                lambda x: tuple([0]) if not isinstance(x.ad, tuple) else tuple(np.array(x.ad) / x.dp), axis=1
            )
        df["max_vaf"] = df["vaf"].apply(lambda x: 0 if isinstance(x, float) else max(x))
        if "qual" not in df or (~df.qual.isna()).sum() == 0:
            df["qual"] = df["tree_score"]

        genotypes = df["gt_ground_truth"] + df["gt_ultima"]
        df["error_type"] = genotypes.apply(self.get_error_type)
        df.rename(columns={"hmer_indel_length": "hmer_length"}, inplace=True)
        return df

    def load_sv_concordance_df(self) -> tuple[dict, dict]:
        """
        Read a pickle file and convert it to a dictionary of DataFrames.

        Parameters
        ----------
        Returns
        -------
        tuple[dict,dict]
            No GT statistics and GT statistics dictionaries.
        """
        with open(self.concordance_file, "rb") as f:
            data = pickle.load(f)
        dfs_no_gt = dict((k, v) for k, v in data.items() if k.endswith("counts"))
        dfs_with_gt = dict((k, v) for k, v in data.items() if not k.endswith("counts"))

        return dfs_no_gt, dfs_with_gt

    def __get_rename_dict(self):
        if self.reference_version == "hg38":
            return {"LCR-hs38": "LCR"}
        if self.reference_version == "hg19":
            return {
                "LCR-hg19_tab_no_chr": "LCR",
                "mappability.hg19.0_tab_no_chr": "mappability.0",
                "ug_hcr_hg19_no_chr": "ug_hcr",
            }
        return {}

    def __columns_subset(self, exome_column_name):
        common_columns = [
            "indel",
            "hmer_indel_length",
            "tree_score",
            "filter",
            "blacklst",
            "classify",
            "classify_gt",
            "indel_length",
            "hmer_indel_nuc",
            "well_mapped_coverage",
            "base",
            "call",
            "gt_ground_truth",
            "gt_ultima",
            "ad",
            "dp",
            "vaf",
            "ref",
            "alleles",
            exome_column_name,
            "gc_content",
            "indel_classify",
            "qual",
            "gq",
        ]
        if self.reference_version == "hg38":
            return common_columns + ["LCR-hs38", "mappability.0", "ug_hcr", "callable"]

        if self.reference_version == "hg19":
            return common_columns + [
                "LCR-hg19_tab_no_chr",
                "mappability.hg19.0_tab_no_chr",
                "ug_hcr_hg19_no_chr",
                "callable",
            ]

        return common_columns

    @staticmethod
    def get_error_type(genotype_pair: tuple) -> ErrorType:
        gtr_gt = set(genotype_pair[0:2])
        call_gt = set(genotype_pair[2:4])

        if gtr_gt == call_gt:
            return ErrorType.NO_ERROR

        if gtr_gt in ({0}, {None}):
            return ErrorType.NOISE

        if call_gt in ({0}, {None}):
            return ErrorType.NO_VARIANT

        if gtr_gt.intersection(call_gt) == gtr_gt:
            return ErrorType.HOM_TO_HET

        if gtr_gt.intersection(call_gt) == call_gt:
            return ErrorType.HET_TO_HOM

        return ErrorType.WRONG_ALLELE
