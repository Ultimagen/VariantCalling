import filecmp
from os.path import join as pjoin
from test import get_resource_dir, test_dir

from ugvc.pipelines.mrd.create_control_signature import create_control_signature

inputs_dir = get_resource_dir(__file__)
general_inputs_dir = pjoin(test_dir, "resources", "general")


def test_create_control_signature(tmpdir):
    reference_fasta = pjoin(general_inputs_dir, "sample.fasta")
    signature = pjoin(
        inputs_dir, "150382-BC04.filtered_signature.chr20_1_100000.vcf.gz",
    )
    control_signature = pjoin(
        inputs_dir, "150382-BC04.filtered_signature.chr20_1_100000.control.vcf.gz",
    )
    output_control_signature = pjoin(tmpdir, "control_signature.vcf.gz")

    create_control_signature(
        signature,
        reference_fasta,
        control_signature_file_output=output_control_signature,
    )
    filecmp.cmp(control_signature, output_control_signature)
