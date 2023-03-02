import numpy as np

from ugvc.comparison.concordance_utils import read_hdf
from ugvc.reports.report_utils import ErrorType


class ReportDataLoader:
    def __init__(self, concordance_file: str, reference_version: str):
        self.concordance_file = concordance_file
        self.reference_version = reference_version
        self.rename_dict = self.__get_rename_dict()

    def load_concordance_df(self):
        df = read_hdf(self.concordance_file, key="all", skip_keys=["concordance", "input_args"])
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
