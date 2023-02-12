import pandas as pd

from ugvc.comparison.concordance_utils import read_hdf


class ReportDataLoader:
    def __init__(self, concordance_file: str, reference_version: str):
        self.concordance_file = concordance_file
        self.reference_version = reference_version
        self.columns_to_select, self.rename_dict = self.__choose_relevant_df_columns()
        self.sources = {}

    def load_concordance_df(self):
        # TODO add description what this table represents?
        self.sources["Trained wo gt"] = (self.concordance_file, "concordance")
        df = read_hdf(self.concordance_file, key="concordance")
        df.rename(columns=self.rename_dict, inplace=True)
        return df

    def load_whole_genome_concordance_df(self):
        """Load the concordance data for the entire genome"""
        self.sources["whole genome"] = (self.concordance_file, "all")
        with pd.HDFStore(self.concordance_file) as hdf:
            keys = hdf.keys()
            wg_dfs = []
            for k in keys:
                if k in {"/concordance", "/input_args"}:
                    continue
                tmp: pd.DataFrame = pd.read_hdf(hdf, key=k)
                tmp = tmp[[x for x in self.columns_to_select if x in tmp.columns]]
                wg_dfs.append(tmp)
            wg_df = pd.concat(wg_dfs)
            wg_df.rename(columns=self.rename_dict, inplace=True)

            return wg_df

    def load_trained_without_gt_concordance_df(self, trained_with_gt):
        self.sources["Trained with gt"] = (trained_with_gt, "scored_concordance")
        df = read_hdf(trained_with_gt, key="scored_concordance")
        df.rename(columns=self.rename_dict, inplace=True)
        return df

    def __choose_relevant_df_columns(self):
        columns_to_select, rename_dict = None, None
        if self.reference_version == "hg38":
            columns_to_select = [
                "indel",
                "hmer_indel_length",
                "tree_score",
                "filter",
                "blacklst",
                "classify",
                "classify_gt",
                "indel_length",
                "hmer_indel_nuc",
                "ref",
                "gt_ground_truth",
                "well_mapped_coverage",
                "mappability.0",
                "ug_hcr",
                "LCR-hs38",
            ]
            rename_dict = {"LCR-hs38": "LCR"}
        elif self.reference_version == "hg19":
            columns_to_select = [
                "indel",
                "hmer_indel_length",
                "tree_score",
                "filter",
                "blacklst",
                "classify",
                "classify_gt",
                "indel_length",
                "hmer_indel_nuc",
                "ref",
                "gt_ground_truth",
                "well_mapped_coverage",
                "LCR-hg19_tab_no_chr",
                "mappability.hg19.0_tab_no_chr",
                "ug_hcr_hg19_no_chr",
            ]
            rename_dict = {
                "LCR-hg19_tab_no_chr": "LCR",
                "mappability.hg19.0_tab_no_chr": "mappability.0",
                "ug_hcr_hg19_no_chr": "ug_hcr",
            }
        return columns_to_select, rename_dict
