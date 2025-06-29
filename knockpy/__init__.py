__version__ = "1.3.5"
__all__ = [
    "constants",
    "dgp",
    "ggm",
    "knockoff_stats",
    "knockoff_filter",
    "knockoffs",
    "mac",
    "metro",
    "mlr",
    "mrc",
    "smatrix",
    "utilities",
    "KnockoffFilter",
]

from . import (
    constants,
    dgp,
    ggm,
    knockoff_filter,
    knockoff_stats,
    knockoffs,
    mac,
    metro,
    mlr,
    mrc,
    smatrix,
    utilities,
)
from .knockoff_filter import KnockoffFilter

name = "knockpy"
