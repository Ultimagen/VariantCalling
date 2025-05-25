from io import StringIO

import pandas as pd
from ugbio_core import stats_utils
from ugbio_core.vcfbed import vcftools

from ugvc.pipelines.sv_stats_collect import (
    collect_size_type_histograms,
    collect_sv_stats,
    concordance_with_gt,
    concordance_with_gt_roc,
)


class TestCollectSizeTypeHistograms:
    # Mock VCF data
    mock_vcf_data = """
        #CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO
        1       1000    .       A       <DEL>   .       PASS    SVLEN=-500;SVTYPE=DEL
        1       2000    .       A       <INS>   .       PASS    SVLEN=300;SVTYPE=INS
        1       3000    .       A       <DEL>   .       PASS    SVLEN=-1000;SVTYPE=DEL
        1       4000    .       A       <INS>   .       PASS    SVLEN=700;SVTYPE=INS
        1       5000    .       A       <DUP>   .       PASS    SVLEN=2000;SVTYPE=DUP
        """

    def test_collect_size_type_histograms(self):
        # read the csv into pandas dataframe vcf_df using whitespace as separator

        vcf_df = pd.read_csv(
            StringIO(self.mock_vcf_data),
            sep=r"\s+",
            comment="#",
            names=["chrom", "pos", "id", "ref", "alt", "qual", "filter", "info"],
        )

        # Mock the `vcftools.get_vcf_df` function
        def mock_get_vcf_df(*args, **kwargs):
            vcf_df["svlen"] = vcf_df["info"].str.extract(r"SVLEN=(-?\d+)").astype(float)
            vcf_df["svtype"] = vcf_df["info"].str.extract(r"SVTYPE=([A-Z]+)")
            return vcf_df

        # Patch the function
        vcftools.get_vcf_df = mock_get_vcf_df

        # Call the function
        result = collect_size_type_histograms("mock_path")

        # Assertions
        assert "type_counts" in result
        assert "length_counts" in result
        assert "length_by_type_counts" in result

        # Check type counts
        expected_type_counts = pd.Series({"DEL": 2, "INS": 2, "DUP": 1})
        pd.testing.assert_series_equal(
            result["type_counts"], expected_type_counts, check_dtype=False, check_names=False
        )

        # Check length counts
        expected_length_counts = pd.Series(
            {
                "50-100": 0,
                "100-300": 0,
                "300-500": 1,
                "0.5-1k": 2,
                "1k-5k": 2,
                "5k-10k": 0,
                "10k-100k": 0,
                "100k-1M": 0,
                ">1M": 0,
            }
        )
        pd.testing.assert_series_equal(
            result["length_counts"],
            expected_length_counts,
            check_names=False,
            check_dtype=False,
            check_index_type=False,
            check_index=False,
        )

        # Check length by type counts
        expected_length_by_type_counts = (
            pd.DataFrame(
                {
                    "50-100": {"DEL": 0, "INS": 0, "DUP": 0},
                    "100-300": {"DEL": 0, "INS": 0, "DUP": 0},
                    "300-500": {"DEL": 0, "INS": 1, "DUP": 0},
                    "0.5-1k": {"DEL": 1, "INS": 1, "DUP": 0},
                    "1k-5k": {"DEL": 1, "INS": 0, "DUP": 1},
                    "5k-10k": {"DEL": 0, "INS": 0, "DUP": 0},
                    "10k-100k": {"DEL": 0, "INS": 0, "DUP": 0},
                    "100k-1M": {"DEL": 0, "INS": 0, "DUP": 0},
                    ">1M": {"DEL": 0, "INS": 0, "DUP": 0},
                }
            )
            .fillna(0)
            .sort_index(axis=0)
        )
        pd.testing.assert_frame_equal(
            result["length_by_type_counts"], expected_length_by_type_counts, check_dtype=False, check_names=False
        )

    def test_collect_size_type_histograms_empty_vcf(self):
        # Mock empty VCF data
        vcf_data = """
        #CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO
        """
        # Read the VCF data, ensuring it handles empty content
        vcf_df = pd.read_csv(
            StringIO(vcf_data.strip()),  # Strip leading/trailing whitespace
            sep="\t",
            comment="#",
            names=["chrom", "pos", "id", "ref", "alt", "qual", "filter", "info"],
            skiprows=1,  # Skip the header line to avoid parsing issues
        )

        # Ensure the DataFrame has the required columns
        if vcf_df.empty:
            vcf_df = pd.DataFrame(
                columns=["chrom", "pos", "id", "ref", "alt", "qual", "filter", "info", "svlen", "svtype"]
            )

        # Mock the `vcftools.get_vcf_df` function
        def mock_get_vcf_df(*args, **kwargs):
            return vcf_df

        # Patch the function
        vcftools.get_vcf_df = mock_get_vcf_df

        # Call the function
        result = collect_size_type_histograms("mock_path")

        # Assertions
        assert result["type_counts"].empty
        # Check that length_counts contains only zero counts
        assert (result["length_counts"] == 0).all()
        # Check that length_by_type_counts contains only zero counts
        assert (result["length_by_type_counts"] == 0).all().all()

    def test_concordance_with_gt(self):
        # Mock data
        df_base = pd.DataFrame({"label": ["TP", "TP", "FN", "FN"]})
        df_calls = pd.DataFrame({"label": ["TP", "TP", "FP", "FP"]})

        # Call the function
        result = concordance_with_gt(df_base, df_calls)

        # Assertions
        assert result["TP_base"] == 2
        assert result["TP_calls"] == 2
        assert result["FN"] == 2
        assert result["FP"] == 2
        assert result["Precision"] == 0.5
        assert result["Recall"] == 0.5
        assert result["F1"] == 0.5

    def test_concordance_with_gt_roc(self):
        # Mock data
        df_base = pd.DataFrame({"label": ["TP", "FN", "FN"], "qual": [0.9, 0.8, 0.7]})
        df_calls = pd.DataFrame({"label": ["TP", "FP", "FP"], "qual": [0.95, 0.6, 0.5]})

        # Mock `stats_utils.precision_recall_curve`
        def mock_precision_recall_curve(gt, predictions, fn_mask, pos_label, min_class_counts_to_output):
            return [0.8, 0.9], [0.7, 0.8], [0.5, 0.6], []

        stats_utils.precision_recall_curve = mock_precision_recall_curve

        # Call the function
        df = concordance_with_gt_roc(df_base, df_calls)
        precision = df["precision"]
        recall = df["recall"]
        thresholds = df["thresholds"]

        # Assertions
        assert precision == [0.8, 0.9]
        assert recall == [0.7, 0.8]
        assert thresholds == [0.5, 0.6]

    def test_collect_sv_stats_without_concordance(self):

        # Mock the `vcftools.get_vcf_df` function
        def mock_get_vcf_df(*args, **kwargs):
            vcf_df = pd.read_csv(
                StringIO(self.mock_vcf_data),
                sep=r"\s+",
                comment="#",
                names=["chrom", "pos", "id", "ref", "alt", "qual", "filter", "info"],
            )
            vcf_df["svlen"] = vcf_df["info"].str.extract(r"SVLEN=(-?\d+)").astype(float)
            vcf_df["svtype"] = vcf_df["info"].str.extract(r"SVTYPE=([A-Z]+)")
            return vcf_df

        vcftools.get_vcf_df = mock_get_vcf_df

        # Call the function
        sv_stats, concordance_stats, fp_stats = collect_sv_stats("mock_path")

        # Assertions
        assert "type_counts" in sv_stats
        assert "length_counts" in sv_stats
        assert "length_by_type_counts" in sv_stats
        assert concordance_stats == {}
        assert isinstance(fp_stats, pd.Series) and fp_stats.empty

    def test_collect_sv_stats_with_concordance(self):
        # Mock VCF data
        vcf_data = """
        #CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO
        1       1000    .       A       <DEL>   10       PASS    SVLEN=-500;SVTYPE=DEL
        1       2000    .       A       <INS>   11       PASS    SVLEN=300;SVTYPE=INS
        """
        # Mock concordance HDF5 data
        df_base = pd.DataFrame(
            {"label": ["TP", "FN"], "svlen": [-500, 300], "svtype": ["DEL", "INS"], "qual": [10, 11]}
        )
        df_calls = pd.DataFrame(
            {"label": ["TP", "FP"], "svlen": [-500, 300], "svtype": ["DEL", "INS"], "qual": [10, 11]}
        )

        # Mock the `vcftools.get_vcf_df` function
        def mock_get_vcf_df(*args, **kwargs):
            vcf_df = pd.read_csv(
                StringIO(vcf_data),
                sep=r"\s+",
                comment="#",
                names=["chrom", "pos", "id", "ref", "alt", "qual", "filter", "info"],
            )
            vcf_df["svlen"] = vcf_df["info"].str.extract(r"SVLEN=(-?\d+)").astype(float)
            vcf_df["svtype"] = vcf_df["info"].str.extract(r"SVTYPE=([A-Z]+)")
            return vcf_df

        vcftools.get_vcf_df = mock_get_vcf_df

        # Mock `pd.read_hdf`
        def mock_read_hdf(filepath, key):
            if key == "base":
                return df_base
            elif key == "calls":
                return df_calls

        pd.read_hdf = mock_read_hdf

        # Mock `stats_utils.precision_recall_curve`
        def mock_precision_recall_curve(gt, predictions, fn_mask, pos_label, min_class_counts_to_output):
            return [0.8, 0.9], [0.7, 0.8], [0.5, 0.6], []

        stats_utils.precision_recall_curve = mock_precision_recall_curve

        # Call the function
        sv_stats, concordance_stats, fp_stats = collect_sv_stats("mock_path", "mock_concordance.h5")

        # Assertions
        assert "type_counts" in sv_stats
        assert "length_counts" in sv_stats
        assert "length_by_type_counts" in sv_stats
        assert "ALL_concordance" in concordance_stats
        assert "ALL_roc" in concordance_stats
        assert concordance_stats["ALL_concordance"]["TP_base"] == 1
        assert concordance_stats["ALL_concordance"]["TP_calls"] == 1
        assert concordance_stats["ALL_concordance"]["FN"] == 1
        assert concordance_stats["ALL_concordance"]["FP"] == 1
        assert concordance_stats["ALL_roc"]["precision"] == [0.8, 0.9]
        assert concordance_stats["ALL_roc"]["recall"] == [0.7, 0.8]
        assert concordance_stats["ALL_roc"]["thresholds"] == [0.5, 0.6]
        assert isinstance(fp_stats, pd.Series) and fp_stats.values[0] == 1
