from time import time


def timer(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t_start = time()
        result = func(*args, **kwargs)
        t_end = time()
        print(f"Function {func.__name__!r} executed in {(t_start - t_end):.4f}s")
        return result

    return wrap_func
