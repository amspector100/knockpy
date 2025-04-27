import ctypes


def srand(val: int) -> None:
    libc = ctypes.CDLL(None)
    libc.srand.argtypes = [ctypes.c_uint]
    libc.srand(val)
