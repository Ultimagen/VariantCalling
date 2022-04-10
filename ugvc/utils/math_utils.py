def safe_divide(a: float, b: float, return_if_b_is_0: int = 0):
    """

    Parameters
    ----------
    a : first float
    b : second float
    return_if_b_is_0 : return value whenever b is 0 (undivisable)

    Returns
    -------
    if b != 0 -> a/b
    else -> return_if_b_is_0
    """
    if b == 0:
        return return_if_b_is_0
    else:
        return a / b
