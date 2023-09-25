import pytest
from simppl.simple_pipeline import SimplePipeline

from ugvc.vcfbed.filter_bed import intersect_bed_regions


@pytest.mark.parametrize(
    "include_regions, exclude_regions, expected_output",
    [
        # Test with a single include region and no exclude regions
        (["chr1\t10\t20\n"], None, ["chr1\t10\t20\n"]),
        # Test with multiple include regions and no exclude regions (they overlap)
        (["chr1\t10\t20\n", "chr1\t15\t25\n"], None, ["chr1\t15\t20\n"]),
        # Test with include and exclude regions (overlap partially)
        (["chr1\t10\t20\n"], ["chr1\t15\t25\n"], ["chr1\t10\t15\n"]),
        # Test with multiple overlapping include regions and overlapping exclude regions
        (
            ["chr1\t10\t30\n", "chr1\t20\t40\n"],
            ["chr1\t25\t35\n"],
            ["chr1\t20\t25\n"],
        ),
        # Test where exclude regions completely remove include regions
        (
            ["chr1\t10\t20\n", "chr2\t30\t40\n"],
            ["chr1\t10\t20\n", "chr2\t30\t40\n"],
            [],
        ),
        # Test with include and exclude regions (overlap partially) - like the next test without headers
        (
            ["chr1\t5\t25\n", "chr1\t15\t40\n"],
            ["chr1\t10\t20\n"],
            ["chr1\t20\t25\n"],
        ),
        # Test bed files with headers
        (
            ["##header1\nchr1\t5\t25\n", "##header2\nchr1\t15\t40\n"],
            ["##header3\nchr1\t10\t20\n"],
            ["chr1\t20\t25\n"],
        ),
    ],
)
def test_intersect_bed_regions(tmpdir, include_regions, exclude_regions, expected_output):
    include_files = []
    for idx, content in enumerate(include_regions):
        file_path = tmpdir.join(f"include_{idx}.bed")
        file_path.write(content)
        include_files.append(str(file_path))

    exclude_files = None
    if exclude_regions:
        exclude_files = []
        for idx, content in enumerate(exclude_regions):
            file_path = tmpdir.join(f"exclude_{idx}.bed")
            file_path.write(content)
            exclude_files.append(str(file_path))

    output_file = str(tmpdir.join("output.bed"))

    intersect_bed_regions(
        include_regions=include_files,
        exclude_regions=exclude_files,
        output_bed=output_file,
        sp=SimplePipeline(0, 100),
    )

    with open(output_file, "r") as f:
        result = f.readlines()

    assert result == expected_output
