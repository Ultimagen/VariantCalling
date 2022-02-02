from ugvc.utils.math import safe_divide


def get_precision(fp: int, tp: int) -> float:
    return 1 - safe_divide(fp, fp + tp)

def get_recall(fn: int, tp: int) -> float:
    return 1 - safe_divide(fn,  fn + tp)

def get_f1(precision: float, recall: float) -> float:
    return safe_divide(2 * precision * recall, precision + recall)