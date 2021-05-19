import warnings
import numpy as np
import scipy as sp
from scipy import stats
import scipy.linalg

from .constants import DEFAULT_TOL
from . import utilities, mrc, constants

# For SDP
import time
import cvxpy as cp
try:
    from pydsdp.dsdp5 import dsdp
    DSDP_AVAILABLE = True
except:
    DSDP_AVAILABLE = False
try:
    import choldate
    CHOLDATE_AVAILABLE = True
except:
    CHOLDATE_AVAILABLE = False

# Options for group SDP solver
OBJECTIVE_OPTIONS = ["abs", "pnorm", "norm"]


def TestIfCorrMatrix(Sigma):
    p = Sigma.shape[0]
    diag = np.diag(Sigma)
    if np.sum(np.abs(diag - np.ones(p))) > p * 1e-2:
        raise ValueError("Sigma is not a correlation matrix. Scale it properly first.")


def calc_min_group_eigenvalue(Sigma, groups, tol=DEFAULT_TOL, verbose=False):
    """
    Calculates the minimum "group" eigenvalue of a covariance 
    matrix Sigma: see Dai and Barber 2016. This is useful for
    constructing equicorrelated (group) knockoffs.
    """

    # Test corr matrix
    TestIfCorrMatrix(Sigma)

    # Construct group block matrix apprx of Sigma
    p = Sigma.shape[0]
    D = np.zeros((p, p))
    for j in np.unique(groups):

        # Select subset of cov matrix
        inds = np.where(groups == j)[0]
        full_inds = np.ix_(inds, inds)
        group_sigma = Sigma[full_inds]

        # Take square root of inverse
        inv_group_sigma = utilities.chol2inv(group_sigma)
        sqrt_inv_group_sigma = sp.linalg.sqrtm(inv_group_sigma)

        # Fill in D
        D[full_inds] = sqrt_inv_group_sigma

    # Test to make sure this is positive definite
    min_d_eig = np.linalg.eigh(D)[0].min()
    if min_d_eig < -1 * tol:
        raise ValueError(f"Minimum eigenvalue of block matrix D is {min_d_eig}")

    # Find minimum eigenvalue
    DSig = np.dot(D, Sigma)
    DSigD = np.dot(DSig, D)
    gamma = min(2 * np.linalg.eigh(DSigD)[0].min(), 1)

    # Warn if imaginary
    if np.imag(gamma) > tol:
        warnings.warn(
            "The minimum eigenvalue is not real, is the cov matrix pos definite?"
        )
    gamma = np.real(gamma)

    return gamma


def solve_equicorrelated(Sigma, groups, tol=DEFAULT_TOL, verbose=False, num_iter=10):
    """ Calculates the block diagonal matrix S using the 
    equicorrelated method described by Dai and Barber 2016.

    Parameters
    ----------
    Sigma : np.ndarray
        ``(p, p)``-shaped covariance matrix of X
    groups : np.ndarray
        For group knockoffs, a p-length array of integers from 1 to 
        num_groups such that ``groups[j] == i`` indicates that variable `j`
        is a member of group `i`. Defaults to ``None`` (regular knockoffs). 
    tol : float
        Minimum permissible eigenvalue of 2Sigma - S and S.

    Returns
    -------
    S : np.ndarray
        ``(p, p)``-shaped (block) diagonal matrix used to generate knockoffs
    """

    # Get eigenvalues and decomposition
    p = Sigma.shape[0]
    if groups is None:
        groups = np.arange(1, p + 1, 1)
    if np.all(groups == np.arange(1, p + 1, 1)):
        gamma = min(2 * np.linalg.eigh(Sigma)[0].min(), 1)
        S = gamma * np.eye(p)
    else:
        gamma = calc_min_group_eigenvalue(Sigma, groups, tol=tol, verbose=verbose)

        # Start to fill up S
        S = np.zeros((p, p))
        for j in np.unique(groups):

            # Select subset of cov matrix
            inds = np.where(groups == j)[0]
            full_inds = np.ix_(inds, inds)
            group_sigma = Sigma[full_inds]

            # fill up S
            S[full_inds] = gamma * group_sigma

    # Scale to make this PSD using binary search
    S, _ = utilities.scale_until_PSD(Sigma, S, tol, num_iter)

    return S


def solve_SDP(Sigma, verbose=False, num_iter=10, tol=DEFAULT_TOL):
    """ 
    Solves ungrouped SDP to create S-matrix for MAC-minimizing knockoffs.

    Parameters
    ----------
    Sigma : np.ndarray
        ``(p, p)``-shaped covariance matrix of X
    verbose : bool
        If True, prints updates during optimization.
    num_iter : int
        Number of iterations in a final binary search to account for
        numerical errors and ensure 2Sigma - S is PSD.
    tol : float
        Minimum permissible eigenvalue of 2Sigma - S and S.

    Returns
    -------
    S : np.ndarray
        ``(p, p)``-shaped diagonal S-matrix used to generate knockoffs
    """

    # Note this DSDP solver is fast but its input format is confusing.
    # This solves:
    # minimize c^T y s.t.
    # Ay <= b
    # F0 + y1 F1 + ... + yp Fp > 0 where F0,...Fp are PSD matrices
    # However the variables here do NOT correspond to the variables
    # in the equations because the Sedumi format is strange -
    # see https://www.ece.uvic.ca/~wslu/Talk/SeDuMi-Remarks.pdf
    # Also, the "l" argument in the K options dictionary
    # in the SDP package may not work.
    # TODO: make this work for group SDP.
    # Idea: basically, add more variables for the off-diagonal elements
    # and maximize their sum subject to the constraint that they can't
    # be larger than the corresponding off-diagonal elements of Sigma
    # (I.e. make the linear constraints larger...)

    # This function requires DSDP.
    # The group_SDP formulation does not.
    if not DSDP_AVAILABLE:
        return solve_group_SDP(
            Sigma=Sigma,
            verbose=verbose,
            num_iter=num_iter,
            tol=tol,
        )

    # Constants
    TestIfCorrMatrix(Sigma)
    p = Sigma.shape[0]
    maxtol = np.linalg.eigh(Sigma)[0].min() / 10
    if tol > maxtol and verbose:
        warnings.warn(
            f"Reducing SDP tol from {tol} to {maxtol}, otherwise SDP would be infeasible"
        )
    tol = min(maxtol, tol)

    # Construct C (-b + vec(F0) from above)
    # Note the tolerance here prevents the min. val
    # of S from being too small.
    Cl1 = np.diag(-1 * tol * np.ones(p)).reshape(1, p ** 2)
    Cl2 = np.diag(np.ones(p)).reshape(1, p ** 2)
    Cs = np.reshape(2 * Sigma, [1, p * p])
    C = np.concatenate([Cl1, Cl2, Cs], axis=1)

    # Construct A
    rows = []
    cols = []
    data = []
    for j in range(p):
        rows.append(j)
        cols.append((p + 1) * j)
        data.append(-1)
    Al1 = sp.sparse.csr_matrix((data, (rows, cols)))
    Al2 = -1 * Al1.copy()
    As = Al2.copy()
    A = sp.sparse.hstack([Al1, Al2, As])

    # Construct b
    b = np.ones([p, 1])

    # Options
    K = {}
    K["s"] = [p, p, p]
    OPTIONS = {
        "gaptol": 1e-6,
        "maxit": 1000,
        "logsummary": 1 if verbose else 0,
        "outputstats": 1 if verbose else 0,
        "print": 1 if verbose else 0,
    }

    # Solve
    warnings.filterwarnings("ignore")
    result = dsdp(A, b, C, K, OPTIONS=OPTIONS)
    warnings.resetwarnings()

    # Raise an error if unsolvable
    status = result["STATS"]["stype"]
    if status != "PDFeasible":
        raise ValueError(f"DSDP solver returned status {status}, should be PDFeasible")
    S = np.diag(result["y"])

    # Scale to make this PSD using binary search
    S, gamma = utilities.scale_until_PSD(Sigma, S, tol, num_iter)
    if verbose:
        mineig = np.linalg.eigh(2 * Sigma - S)[0].min()
        print(
            f"After SDP, mineig is {mineig} after {num_iter} line search iters."
        )

    return S


def solve_group_SDP(
    Sigma,
    groups=None,
    verbose=False,
    objective="abs",
    norm_type=2,
    num_iter=10,
    tol=DEFAULT_TOL,
    dsdp_warning=True,
    **kwargs,
):
    """
    Solves the MAC-minimizng SDP formulation for group knockoffs:
    extends Barer and Candes 2015/ Candes et al 2018.

    Sigma : np.ndarray
        ``(p, p)``-shaped covariance matrix of X
    groups : np.ndarray
        For group knockoffs, a p-length array of integers from 1 to 
        num_groups such that ``groups[j] == i`` indicates that variable `j`
        is a member of group `i`. Defaults to ``None`` (regular knockoffs). 
    verbose : bool
        If True, prints updates during optimization.
    objective : str
        How to optimize the S matrix for group knockoffs. 
        There are several options:
        - 'abs': minimize sum(abs(Sigma - S))
        - 'pnorm': minimize Lp-th matrix norm.
        - 'norm': minimize different type of matrix norm
        (see norm_type below).
    norm_type : str or int
        - When objective == 'pnorm', a float specifying which Lp-th matrix norm
        to use. Can be any float >= 1. 
        - When objective == 'norm', can be 'fro', 'nuc', np.inf, or 1.
    num_iter : int
        Number of iterations in a final binary search to account for
        numerical errors and ensure 2Sigma - S is PSD.
    tol : float
        Minimum permissible eigenvalue of 2Sigma - S and S.
    kwargs : dict
        Keyword arguments to pass to the ``cvxpy.Problem.solve()`` method.

    Returns
    -------
    S : np.ndarray
        ``(p, p)``-shaped (block) diagonal matrix used to generate knockoffs
    """

    # By default we lower the convergence epsilon a bit for drastic speedup.
    if "eps" not in kwargs:
        kwargs["eps"] = 5e-3

    # Default groups
    p = Sigma.shape[0]
    if groups is None:
        groups = np.arange(1, p + 1, 1)

    # Test corr matrix
    TestIfCorrMatrix(Sigma)

    # Check to make sure the objective is valid
    objective = str(objective).lower()
    if objective not in OBJECTIVE_OPTIONS:
        raise ValueError(f"Objective ({objective}) must be one of {OBJECTIVE_OPTIONS}")
    # Warn user if they're using a weird norm...
    if objective == "norm" and norm_type == 2:
        warnings.warn(
            "Using norm objective and norm_type = 2 can lead to strange behavior: consider using Frobenius norm"
        )
    # Find minimum tolerance, possibly warn user if lower than they specified
    maxtol = np.linalg.eigh(Sigma)[0].min() / 1.1
    if tol > maxtol and verbose:
        warnings.warn(
            f"Reducing SDP tol from {tol} to {maxtol}, otherwise SDP would be infeasible"
        )
    tol = min(maxtol, tol)

    # Figure out sizes of groups
    m = groups.max()
    group_sizes = utilities.calc_group_sizes(groups)

    # Possibly solve non-grouped SDP
    if m == p:
        if DSDP_AVAILABLE:
            return solve_SDP(Sigma=Sigma, verbose=verbose, num_iter=num_iter, tol=tol,)
        elif dsdp_warning:
            warnings.warn(constants.DSDP_WARNING)

    # Sort the covariance matrix according to the groups
    inds, inv_inds = utilities.permute_matrix_by_groups(groups)
    sortedSigma = Sigma[inds][:, inds]

    # Create blocks of semidefinite matrix S,
    # as well as the whole matrix S
    variables = []
    constraints = []
    S_rows = []
    shift = 0
    for j in range(m):

        # Create block variable
        gj = int(group_sizes[j])
        Sj = cp.Variable((gj, gj), symmetric=True)
        constraints += [Sj >> 0]
        variables.append(Sj)

        # Create row of S
        if shift == 0 and shift + gj < p:
            rowj = cp.hstack([Sj, cp.Constant(np.zeros((gj, p - gj)))])
        elif shift + gj < p:
            rowj = cp.hstack(
                [
                    cp.Constant(np.zeros((gj, shift))),
                    Sj,
                    cp.Constant(np.zeros((gj, p - gj - shift))),
                ]
            )
        elif shift + gj == p and shift > 0:
            rowj = cp.hstack([cp.Constant(np.zeros((gj, shift))), Sj])
        elif gj == p and shift == 0:
            rowj = cp.hstack([Sj])

        else:
            raise ValueError(
                f"shift ({shift}) and gj ({gj}) add up to more than p ({p})"
            )
        S_rows.append(rowj)

        # Incremenet shift
        shift += gj

    # Construct S and Grahm Matrix
    S = cp.vstack(S_rows)
    sortedSigma = cp.Constant(sortedSigma)
    constraints += [2 * sortedSigma - S >> 0]

    # Construct optimization objective
    if objective == "abs":
        objective = cp.Minimize(cp.sum(cp.abs(sortedSigma - S)))
    elif objective == "pnorm":
        objective = cp.Minimize(cp.pnorm(sortedSigma - S, norm_type))
    elif objective == "norm":
        objective = cp.Minimize(cp.norm(sortedSigma - S, norm_type))
    # Note we already checked objective is one of these values earlier

    # Construct, solve the problem.
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=verbose, **kwargs)
    if verbose:
        print("Finished solving SDP!")

    # Unsort and get numpy
    S = S.value
    if S is None:
        raise ValueError(
            "SDP formulation is infeasible. Try decreasing the tol parameter."
        )
    S = S[inv_inds][:, inv_inds]

    # Clip 0 and 1 values
    for i in range(p):
        S[i, i] = max(tol, min(1 - tol, S[i, i]))

    # Scale to make this PSD using binary search
    S, gamma = utilities.scale_until_PSD(Sigma, S, tol, num_iter)

    # Return unsorted S value
    return S
