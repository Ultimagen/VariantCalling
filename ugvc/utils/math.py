
def safe_divide(a: float, b: float, return_if_b_is_0: int = 0):
    if b == 0:
        return return_if_b_is_0
    else:
        return a / b
