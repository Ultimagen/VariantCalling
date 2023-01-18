from __future__ import annotations

class Genotype:
    def __init__(self, genotype: str):
        """
        @param genotype - / separated genotype indices. e.g '0/0', '0/1', '0', '1'
        """
        self.genotype_str = genotype

    def convert_to_reference(self) -> str:
        return "/".join(["0" for _ in self.genotype_str.split("/")])

    def is_reference(self):
        return all(a == "0" for a in self.genotype_str.split("/"))


def sort_gt(gt: tuple[int] | str) -> tuple[int]:
    gt_list = []
    if isinstance(gt, tuple):
        gt_list = list(gt)
    elif isinstance(gt, str):
        gt_list = [int(a) for a in gt.replace("/", "|").split("|")]

    gt_list.sort(key=lambda e: (e is None, e))
    return tuple(gt_list)


def different_gt(gt1: tuple[int] | str, gt2: tuple[int] | str) -> bool:
    out = sort_gt(gt1) != sort_gt(gt2)
    return out

