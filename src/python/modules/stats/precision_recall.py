def get_precision(fp: int, tp: int) -> float:
    return 1 - fp / (fp + tp)

def get_recall(fn: int, tp: int) -> float:
    return 1 - fn / (fn + tp)

def get_f1(precision: float, recall: float) -> float:
    return (2 * precision * recall) /  (precision + recall)