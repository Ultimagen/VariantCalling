import os
from test import get_resource_dir, test_dir

from ugvc.dna.format import DEFAULT_FLOW_ORDER
from ugvc.pipelines import train_models_pipeline


class TestRunTraining:
    inputs_dir = get_resource_dir(__file__)
    general_inputs_dir = f"{test_dir}/resources/general/chr1_head"

    def test_run_training_no_gt(self, tmpdir):
        train_models_pipeline.run(
            [
                "--input_file",
                f"{self.inputs_dir}/180801-UGAv3-26.intervals_annotated.chr1_1_5000000.vcf.gz",
                "--reference",
                f"{self.general_inputs_dir}/Homo_sapiens_assembly38.fasta",
                "--runs_intervals",
                f"{self.general_inputs_dir}/hg38_runs.conservative.bed",
                "--evaluate_concordance",
                "--apply_model",
                "rf_model_ignore_gt_incl_hpol_runs",
                "--blacklist",
                f"{self.inputs_dir}/blacklist_chr1_1_5000000.h5",
                "--flow_order",
                DEFAULT_FLOW_ORDER,
                "--exome_weight",
                "100",
                "--exome_weight_annotation",
                "exome.twist",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/LCR-hs38.bed",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/exome.twist.bed",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/mappability.0.bed",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/hmers_7_and_higher.bed",
                "--output_file_prefix",
                f"{tmpdir}/180801.model",
            ]
        )
        assert os.path.exists(f"{tmpdir}/180801.model.h5")
        assert os.path.exists(f"{tmpdir}/180801.model.pkl")

    def test_run_training_with_gt(self, tmpdir):
        train_models_pipeline.run(
            [
                "--input_file",
                f"{self.inputs_dir}/180801-UGAv3-26_chr1_1_5000000.comp.h5",
                "--input_interval",
                f"{self.inputs_dir}/chr1_1_5000000.bed",
                "--reference",
                f"{self.general_inputs_dir}/Homo_sapiens_assembly38.fasta",
                "--evaluate_concordance",
                "--apply_model",
                "threshold_model_ignore_gt_incl_hpol_runs",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/LCR-hs38.bed",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/exome.twist.bed",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/mappability.0.bed",
                "--annotate_intervals",
                f"{self.general_inputs_dir}/hmers_7_and_higher.bed",
                "--output_file_prefix",
                f"{tmpdir}/180801.model",
            ]
        )
        assert os.path.exists(f"{tmpdir}/180801.model.h5")
        assert os.path.exists(f"{tmpdir}/180801.model.pkl")
