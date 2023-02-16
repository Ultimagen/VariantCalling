from ugvc.comparison.concordance_utils import read_hdf


class ReportDataLoader:
    def __init__(self, concordance_file: str, reference_version: str):
        self.concordance_file = concordance_file
        self.reference_version = reference_version
        self.rename_dict = self.__get_rename_dict()

    def load_concordance_df(self):
        df = read_hdf(self.concordance_file, key="concordance")
        df.rename(columns=self.rename_dict, inplace=True)
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
