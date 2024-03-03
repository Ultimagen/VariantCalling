import os
from test import get_resource_dir

from ugvc.pipelines import train_models_pipeline


class TestRunTraining:
    inputs_dir = get_resource_dir(__file__)

    def test_run_training_no_gt(self, tmpdir):
        train_models_pipeline.run(
            [
                "--train_dfs",
                f"{self.inputs_dir}/train_model_approximate_gt_input.h5",
                "--test_dfs",
                f"{self.inputs_dir}/train_model_approximate_gt_input.h5",
                "--output_file_prefix",
                f"{tmpdir}/approximate_gt.model",
                "--gt_type",
                "approximate",
                "--verbosity",
                "DEBUG",
            ]
        )
        assert os.path.exists(f"{tmpdir}/approximate_gt.model.h5")
        assert os.path.exists(f"{tmpdir}/approximate_gt.model.pkl")
