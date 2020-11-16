""" Numerical constants for the whole package plus docstring constants. """

# Default regularization values for cross-validated
# penalized regressions.
import numpy as np

DEFAULT_REG_VALS = np.logspace(-4, 4, base=10, num=20)
# Default tolerance for mrc/mac solvers.
DEFAULT_TOL = 1e-5
# For line search for mrc/mac solvers
GAMMA_VALS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.9999, 1]