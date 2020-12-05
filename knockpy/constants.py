""" Numerical constants for the whole package plus docstring constants. """

# Default regularization values for cross-validated
# penalized regressions.
import numpy as np

DEFAULT_REG_VALS = np.logspace(-4, 4, base=10, num=20)
# Default tolerance for mrc/mac solvers.
DEFAULT_TOL = 1e-5
# Tolerance for metro (where numerical stability is an issue)
METRO_TOL = 1e-2
# For line search for mrc/mac solvers
GAMMA_VALS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.9999, 1]
# DSDP warning
DSDP_WARNING = f"""
	---------------------------------------------------------
	---------------------------------------------------------
	The scikit-dsdp package is not installed:
	solving an SDP without it will be roughly
	10-100x slower. For installation instructions, see
	https://amspector100.github.io/knockpy/installation.html.
	To disable this warning, set dsdp_warning=False as an 
	argument or a knockoff_kwarg.
	----------------------------------------------------------
	----------------------------------------------------------
"""
# choldate warning
CHOLDATE_WARNING = f"""
	---------------------------------------------------------
	---------------------------------------------------------
	The choldate package is not installed:
	solving for MVR or maxent knockoffs without it will be 
	~5x slower. For instructions on how to install choldate,
	see https://amspector100.github.io/knockpy/installation.html.
	To disable this warning, set choldate_warning=False as an 
	argument or a knockoff_kwarg.
	----------------------------------------------------------
	----------------------------------------------------------
"""