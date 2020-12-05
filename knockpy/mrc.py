""" Methods for minimum-reconstructability knockoffs."""

import warnings
import time
import numpy as np
import scipy as sp
from scipy import stats
from . import utilities, constants
try:
    import choldate
    CHOLDATE_AVAILABLE = True
except:
    CHOLDATE_AVAILABLE = False

def cholupdate(R, x, add=True):
    """
    Performs rank one updates to a cholesky factor `R` in place.
    
    Parameters
    ----------
    R : np.ndarray
        A ``(p,p)``-shaped upper-triangular matrix.
    x : np.ndarray
        A ``(p,)``-shaped vector.
    add : bool
        If True, performs a rank one update; else performs a
        rank one downdate.

    Returns
    -------
    R : np.ndarray
        Suppose the parameter R was a cholesky factor of a matrix V.
        Upon return, R is the cholesky factor of 
        ``V +/- np.outer(x, x)``.

    Notes
    -----
    - This function modifies both ``R`` and ``x`` in place. 
    - The ``choldate`` package is a much faster and more 
    numerically stable alternative.

    """
    p = np.size(x)
    x = x.T
    for k in range(p):
        if add:
              r = np.sqrt(R[k,k]**2 + x[k]**2)
        else:
              r = np.sqrt(R[k,k]**2 - x[k]**2)
        c = r/R[k,k]
        s = x[k]/R[k,k]
        R[k,k] = r
        if add:
            R[k,k+1:p] = (R[k,k+1:p] + s*x[k+1:p])/c
        else:
            R[k,k+1:p] = (R[k,k+1:p] - s*x[k+1:p])/c
        x[k+1:p]= c*x[k+1:p] - s*R[k, k+1:p]
    return R

### Coordinate Descent Solvers
def mvr_loss(Sigma, S, smoothing=0):
    """ 
    Computes minimum variance-based reconstructability
    loss for knockoffs, e.g., the trace of the feature-knockoff
    precision matrix.

    Parameters
    ----------
    Sigma : np.ndarray
        ``(p, p)``-shaped covariance matrix of X
    S : np.ndarray
        ``(p, p)``-shaped S-matrix used to generate knockoffs
    smoothing : float
        Add ``smoothing`` to all eigenvalues of the feature-knockoff
        precision matrix before inverting to avoid numerical
        instability. Defaults to 0.

    Returns
    -------
    loss : float
        The MVR loss for Sigma and S.
    """

    # Inverse of eigenvalues
    trace_invG = (1 / (np.linalg.eigh(2 * Sigma - S)[0] + smoothing)).sum()
    trace_invG = trace_invG + (1 / (np.diag(S) + smoothing)).sum()
    return trace_invG


def maxent_loss(Sigma, S, smoothing=0):
    """
    Computes the log determinant of the feature-knockoff precision
    matrix, which is proportional to the negative entropy of [X, tilde{X}].

    Parameters
    ----------
    Sigma : np.ndarray
        ``(p, p)``-shaped covariance matrix of X
    S : np.ndarray
        ``(p, p)``-shaped S-matrix used to generate knockoffs
    smoothing : float
        Add ``smoothing`` to all eigenvalues of the feature-knockoff
        precision matrix before taking the log determinant
        to avoid numerical instability. Defaults to 0.

    Returns
    -------
    loss : float
        The MMI loss for Sigma and S.
    """
    p = Sigma.shape[0]
    detG = np.log(np.linalg.det(2 * Sigma - S + smoothing * np.eye(p)))
    detG = detG + np.log(np.diag(S + smoothing)).sum()
    return -1 * detG

def mmi_loss(*args, **kwargs):
    """
    Computes the log determinant of the feature-knockoff precision
    matrix, which is proportional mutual information between X and knockoffs.

    This is identical to ``maxent_loss`` and exists only for backwards 
    compatability.
    """
    return maxent_loss(*args, **kwargs)


def solve_mvr(
    Sigma,
    tol=1e-5,
    verbose=False,
    num_iter=10,
    smoothing=0,
    rej_rate=0,
    converge_tol=1,
    choldate_warning=True,
):
    """
    Computes S-matrix used to generate minimum variance-based
    reconstructability knockoffs using coordinate descent.

    Parameters
    ----------
    Sigma : np.ndarray
        ``(p, p)``-shaped covariance matrix of X
    tol : float
        Minimum permissible eigenvalue of 2Sigma - S and S.
    verbose : bool
        If True, prints updates during optimization.
    num_iter : int
        The number of coordinate descent iterations. Defaults to 50.
    smoothing : float
        Add ``smoothing`` to all eigenvalues of the feature-knockoff
        precision matrix before inverting to avoid numerical
        instability. Defaults to 0.
    converge_tol : float
        A parameter specifying the criteria for convergence.
    choldate_warning : bool
        If True, will warn the user if choldate is not installed. 
        Defaults to True.

    Returns
    -------
    S : np.ndarray
        ``(p, p)``-shaped (block) diagonal matrix used to generate knockoffs
    """

    # Warning if choldate not available
    if not CHOLDATE_AVAILABLE and choldate_warning:
        warnings.warn(constants.CHOLDATE_WARNING)

    # Initial constants
    time0 = time.time()
    V = Sigma  # Shorthand prevents lines from spilling over
    p = V.shape[0]
    inds = np.arange(p)
    loss = np.inf
    acc_rate = 1 - rej_rate
    # Takes a bit longer for rej_rate adjusted to converge
    if acc_rate < 1:
        converge_tol = 1e-2

    # Initialize values
    decayed_improvement = 10
    min_eig = np.linalg.eigh(V)[0].min()
    S = min_eig * np.eye(p)
    L = np.linalg.cholesky(2 * V - S + smoothing * np.eye(p))

    for i in range(num_iter):
        np.random.shuffle(inds)
        for j in inds:
            # 1. Compute coefficients cn and cd
            ej = np.zeros(p)  # jth basis element
            ej[j] = 1
            # 1a. Compute cd
            vd = sp.linalg.solve_triangular(a=L, b=ej, lower=True)
            cd = np.power(vd, 2).sum()
            # 1b. Compute vn
            vn = sp.linalg.solve_triangular(a=L.T, b=vd, lower=False)
            cn = -1 * np.power(vn, 2).sum()

            # 2. Construct quadratic equation
            # We want to minimize 1/(sj + delta) - (delta * cn)/(1 - delta * cd)
            coef2 = -1 * cn - np.power(cd, 2)
            coef1 = 2 * (-1 * cn * (S[j, j] + smoothing) + cd)
            coef0 = -1 * cn * (S[j, j] + smoothing) ** 2 - 1
            orig_options = np.roots(np.array([coef2, coef1, coef0]))

            # 3. Eliminate complex solutions
            options = np.array([delta for delta in orig_options if np.imag(delta) == 0])
            # Eliminate solutions which violate PSD-ness
            upper_bound = 1 / cd
            lower_bound = -1 * S[j, j]
            options = np.array(
                [
                    delta
                    for delta in options
                    if delta < upper_bound and delta > lower_bound
                ]
            )
            if options.shape[0] == 0:
                raise RuntimeError(
                    f"All quadratic solutions ({orig_options}) were infeasible or imaginary"
                )

            # 4. If multiple solutions left (unlikely), pick the smaller one
            losses = 1 / (S[j, j] + options) - (options * cn) / (1 - options * cd)
            if losses[0] == losses.min():
                delta = options[0]
            else:
                delta = options[1]

            # 5. Account for rejections
            if acc_rate < 1:
                extra_space = min(min_eig, 0.02) / (i + 2)  # Helps deal with coord desc
                opt_postrej_value = S[j, j] + delta
                opt_prerej_value = opt_postrej_value / (acc_rate)
                opt_prerej_value = min(
                    S[j, j] + upper_bound - extra_space,
                    max(opt_prerej_value, extra_space),
                )
                delta = opt_prerej_value - S[j, j]

            # Update S and L
            x = np.zeros(p)
            x[j] = np.sqrt(np.abs(delta))
            if delta > 0:
                if CHOLDATE_AVAILABLE:
                    choldate.choldowndate(L.T, x)
                else:
                    cholupdate(L.T, x, add=False)
            else:
                if CHOLDATE_AVAILABLE:
                    choldate.cholupdate(L.T, x)
                else:
                    cholupdate(L.T, x, add=True)

            # Set new value for S
            S[j, j] += delta

        # Check for convergence
        prev_loss = loss
        loss = mvr_loss(V, acc_rate * S, smoothing=smoothing)
        if i != 0:
            decayed_improvement = decayed_improvement / 10 + 9 * (prev_loss - loss) / 10
        if verbose:
            print(
                f"After iter {i} at time {np.around(time.time() - time0,3)}, loss={loss}, decayed_improvement={decayed_improvement}"
            )
        if decayed_improvement < converge_tol:
            if verbose:
                print(f"Converged after iteration {i} with loss={loss}")
            break
    # Ensure validity of solution
    S = utilities.shift_until_PSD(S, tol=tol)
    S, _ = utilities.scale_until_PSD(V, S, tol=tol, num_iter=10)
    return S

def solve_maxent(
    Sigma, 
    tol=1e-5,
    verbose=False,
    num_iter=10,
    converge_tol=1e-4,
    choldate_warning=True,
):
    """
    Computes S-matrix used to generate maximum entropy
    knockoffs using coordinate descent.

    Parameters
    ----------
    Sigma : np.ndarray
        ``(p, p)``-shaped covariance matrix of X
    tol : float
        Minimum permissible eigenvalue of 2Sigma - S and S.
    verbose : bool
        If True, prints updates during optimization.
    num_iter : int
        The number of coordinate descent iterations. Defaults to 50.
    converge_tol : float
        A parameter specifying the criteria for convergence.
    choldate_warning : bool
        If True, will warn the user if choldate is not installed. 
        Defaults to True

    Returns
    -------
    S : np.ndarray
        ``(p, p)``-shaped (block) diagonal matrix used to generate knockoffs
    """

    # Warning if choldate not available
    if not CHOLDATE_AVAILABLE and choldate_warning:
        warnings.warn(constants.CHOLDATE_WARNING)

    # Initial constants
    time0 = time.time()
    V = Sigma  # Shorthand prevents lines from spilling over
    p = V.shape[0]
    inds = np.arange(p)
    loss = np.inf

    # Initialize values
    decayed_improvement = 1
    S = np.linalg.eigh(V)[0].min() * np.eye(p)
    L = np.linalg.cholesky(2 * V - S)

    for i in range(num_iter):
        np.random.shuffle(inds)
        for j in inds:
            diff = 2 * V - S

            # Solve cholesky equation
            tildey = 2 * V[j].copy()
            tildey[j] = 0
            x = sp.linalg.solve_triangular(a=L, b=tildey, lower=True)

            # Use cholesky eq to get new update
            zeta = diff[j, j]
            x22 = np.power(x, 2).sum()
            qinvterm = zeta * x22 / (zeta + x22)

            # Inverse of Qj using SWM formula
            sjstar = (2 * V[j, j] - qinvterm) / 2

            # Rank one update for cholesky
            delta = S[j, j] - sjstar
            x = np.zeros(p)
            x[j] = np.sqrt(np.abs(delta))
            if delta > 0:
                if CHOLDATE_AVAILABLE:
                    choldate.cholupdate(L.T, x)
                else:
                    cholupdate(L.T, x, add=False)
            else:
                if CHOLDATE_AVAILABLE:
                    choldate.choldowndate(L.T, x)
                else:
                    cholupdate(L.T, x, add=True)

            # Set new value for S
            S[j, j] = sjstar

        # Check for convergence
        prev_loss = loss
        loss = mmi_loss(V, S)
        if i != 0:
            decayed_improvement = decayed_improvement / 10 + 9 * (prev_loss - loss) / 10
        if verbose:
            print(
                f"After iter {i} at time {np.around(time.time() - time0,3)}, loss={loss}, decayed_improvement={decayed_improvement}"
            )
        if decayed_improvement < converge_tol:
            if verbose:
                print(f"Converged after iteration {i} with loss={loss}")
            break

    # Ensure validity of solution
    S = utilities.shift_until_PSD(S, tol=tol)
    S, _ = utilities.scale_until_PSD(V, S, tol=tol, num_iter=10)
    return S

def solve_mmi(*args, **kwargs):
    """
    Computes S-matrix used to generate minimum mutual information
    knockoffs. This is identical to ``solve_maxent``
    and exists only for backwards compatability.
    """
    return solve_maxent(*args, **kwargs)

def solve_ciknock(
    Sigma, tol=1e-5, num_iter=10,
):
    """
    Computes S-matrix used to generate conditional independence
    knockoffs.

    Parameters
    ----------
    Sigma : np.ndarray
        ``(p, p)``-shaped covariance matrix of X
    tol : float
        Minimum permissible eigenvalue of 2Sigma - S and S.
    num_iter : int
        The number of iterations in the binary search to ensure
        S is feasible.

    Returns
    -------
    S : np.ndarray
        ``(p, p)``-shaped (block) diagonal matrix used to generate knockoffs

    Notes
    -----
    When the S-matrix corresponding to conditional independence knockoffs
    is not feasible, this computes that S matrix and then does a binary 
    search to find the maximum gamma such that gamma * S is feasible.
    """
    # Compute vanilla S_CI
    S = 1 / (np.diag(np.linalg.inv(Sigma)))
    S = np.diag(S)
    # Ensure validity of solution
    S = utilities.shift_until_PSD(S, tol=tol)
    S, _ = utilities.scale_until_PSD(Sigma, S, tol=tol, num_iter=num_iter)
    return S
