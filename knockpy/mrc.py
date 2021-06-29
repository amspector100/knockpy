""" Methods for minimum-reconstructability knockoffs."""

import warnings
import time
import numpy as np
import scipy as sp
from scipy import stats
from . import utilities, constants, mac
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

### MVR
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
        The MVR loss for Sigma and S. This is infinite if S is not feasible.
    """
    # Check for negative eigenvalues
    eigs_S = np.diag(S)
    eigs_diff = np.linalg.eigh(2 * Sigma - S)[0]
    if np.min(eigs_S) < 0 or np.min(eigs_diff) < 0:
        return np.inf

    # Inverse of eigenvalues
    trace_invG = (1 / (eigs_diff + smoothing)).sum() + (1 / (eigs_S + smoothing)).sum()
    return trace_invG

def _solve_mvr_quadratic(cn, cd, sj, i=None, min_eig=None, acc_rate=1, smoothing=0):
    """
    Solves a quadratic equation to find
    the optimal updates for the MVR S-matrix
    based off of cn and cd.
    See https://arxiv.org/pdf/2011.14625.pdf
    """

    # 1. Construct quadratic equation
    # We want to minimize 1/(sj + delta) - (delta * cn)/(1 - delta * cd)
    coef2 = -1 * cn - np.power(cd, 2)
    coef1 = 2 * (-1 * cn * (sj + smoothing) + cd)
    coef0 = -1 * cn * (sj + smoothing) ** 2 - 1
    orig_options = np.roots(np.array([coef2, coef1, coef0]))

    # 2. Eliminate complex solutions
    options = np.array([delta for delta in orig_options if np.imag(delta) == 0])
    # Eliminate solutions which violate PSD-ness
    upper_bound = 1 / cd
    lower_bound = -1 * sj
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
        
    # 3. If multiple solutions left (unlikely), pick the smaller one
    losses = 1 / (sj + options) - (options * cn) / (1 - options * cd)
    if losses[0] == losses.min():
        delta = options[0]
    else:
        delta = options[1]
        
    # 4. Account for rejections
    if acc_rate < 1:
        extra_space = min(min_eig, 0.05) / (i + 2)  # Helps deal with coord desc
        opt_postrej_value = sj + delta
        opt_prerej_value = opt_postrej_value / (acc_rate)
        opt_prerej_value = min(
            sj + upper_bound - extra_space,
            max(opt_prerej_value, extra_space),
        )
        delta = opt_prerej_value - sj
        
    return delta

def solve_mvr_factored(
    D,
    U,
    tol=1e-5,
    verbose=False,
    num_iter=15,
    converge_tol=1e-4,
):
    """
    Computes S-matrix used to generate mvr knockoffs
    using coordinate descent assuming that
    the covariance matrix follows a factor model.
    This means Sigma = D + UU^T for a p x p diagonal matrix
    D and a p x k matrix U. 

    Parameters
    ----------
    D : np.ndarray
        ``p``-shaped array of diagonal elements.
    U : np.ndarray
        ``(p, k)``-shaped matrix. Usually k << p.
    tol : float
        Minimum permissible eigenvalue of 2Sigma - S and S.
    verbose : bool
        If True, prints updates during optimization.
    num_iter : int
        The number of coordinate descent iterations. Defaults to 50.
    converge_tol : float
        A parameter specifying the criteria for convergence.

    Returns
    -------
    S : np.ndarray
        ``(p, p)``-shaped (block) diagonal matrix used to generate knockoffs
    """

    # Initial constants
    p = D.shape[0]
    k = U.shape[1]
    inds = np.arange(p)
    loss = np.inf

    # TODO: this is Omega(p^3).
    # Maybe use https://www.sciencedirect.com/science/article/pii/S2215016118300062
    Sigma = np.diag(D) + np.dot(U, U.T)
    diag_Sigma = np.diag(Sigma)
    mineig = np.linalg.eigh(Sigma)[0].min()
    if mineig < 0:
        raise ValueError("D + UU^T is not PSD")

    # Initialize values
    time0 = time.time()
    decayed_improvement = 1
    Sdiag = np.zeros(p) + mineig
    # These are k x k matrices
    Q, R = sp.linalg.qr(np.eye(k) + 2*np.dot(U.T / (2*D - Sdiag), U))

    quadtime = 0
    solvetime = 0
    for i in range(num_iter):
        np.random.shuffle(inds)
        for j in inds:

            # 1. Calculate parameters cd/cn
            # (see https://arxiv.org/pdf/2011.14625.pdf)
            b = np.dot(Q.T, U[j].T/(2*D[j] - Sdiag[j]))
            x = sp.linalg.solve_triangular(R, b, lower=False)
            diff_inv_ej =  -2*np.dot(U, x)/(2*D - Sdiag)
            diff_inv_ej[j] = diff_inv_ej[j] + 1/(2*D[j] - Sdiag[j])
            cd = diff_inv_ej[j]
            cn = -1 * np.power(diff_inv_ej, 2).sum()

            # 2. Find optimal update
            delta = _solve_mvr_quadratic(cn=cn, cd=cd, sj=Sdiag[j])
            
            # 3. Rank one update to QR decomp
            muj = U[j].T / (2*D[j] - Sdiag[j])
            c = -delta/(1 - delta/(2*D[j] - Sdiag[j]))
            Q, R = sp.linalg.qr_update(
                Q=Q,
                R=R,
                u=-2*c*muj,
                v=muj,
            )
            # 4. Update S
            Sdiag[j] = Sdiag[j] + delta
            
    return np.diag(Sdiag)


def _solve_mvr_ungrouped(
    Sigma,
    tol=1e-5,
    verbose=False,
    num_iter=50,
    smoothing=0,
    rej_rate=0,
    converge_tol=1e-2,
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

            # 2. Construct/solve quadratic equation
            delta = _solve_mvr_quadratic(
                cn=cn,
                cd=cd,
                sj=S[j,j],
                min_eig=min_eig,
                i=i,
                smoothing=smoothing,
                acc_rate=acc_rate,
            )

            # 3. Update S and L
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

    return S

### Group MVR
def _solve_mvr_quadratic_group(cn_diff, cd_diff, cn_S, cd_S):
    """
    Internal helper function for group mvr.
    Useful for diagonal elements.
    """
    # 1. Construct and solve quadratic equation
    coef0 = cn_diff - cn_S
    coef1 = 2*(cd_S * cn_diff + cn_S * cd_diff)
    coef2 = cn_diff * np.power(cd_S, 2) - cn_S * np.power(cd_diff, 2)
    orig_options = np.roots(np.array([coef2, coef1, coef0]))

    # 2. Eliminate complex solutions and solutions which violate PSD-ness
    options = np.array([delta for delta in orig_options if np.imag(delta) == 0])
    upper_bound = 1 / cd_diff # Ensures 2 Sigma - S is PSD
    lower_bound = -1 / cd_S # Ensures S is PSD
    options = options[(options < upper_bound) & (options > lower_bound)]
    if options.shape[0] == 0:
        raise RuntimeError(
            f"All quadratic solutions ({orig_options}) were infeasible or imaginary"
        )
    elif options.shape[0] > 1:
        raise RuntimeError(
            f"Multiple feasible solutions (options={options}), need the lower bound"
        )
    return options[0]

def _mvr_group_contrib(Q, R, i, j):
    """
    Internal helper function for group mvr.
    Useful for off-diagonal elements.

    Parameters
    ----------
    Q, R : QR decomposition
    i, j: indices for rank-2 update

    Returns
    -------
    f : function
        f(delta) equals
        Tr((QR + delta e_i e_j^T + delta e_j e_i^T)^{-1})
        up to a constant not depending on delta.
    min_delta : int
        Minimum value of delta for which the prior matrix is PSD.
    max_delta : int
        Maximum value of delta for which the prior matrix is PSD.
    """
    
    # Notational note: W = (QR)^{-1}
    Wj = sp.linalg.solve_triangular(a=R, b=Q[j], lower=False)
    Wi = sp.linalg.solve_triangular(a=R, b=Q[i], lower=False)
    def objective(delta):
        num = 2 * np.dot(Wj, Wi)*(1 + delta * Wj[i])
        num -= delta * Wj[j] * np.dot(Wi, Wi)
        num -= delta * Wi[i] * np.dot(Wj, Wj)
        denom = np.power(1 + delta * Wj[i], 2) - np.power(delta, 2) * Wi[i] * Wj[j]
        return delta * num/denom
    
    # Binary search for min / max. value of delta
    I = np.eye(2)
    B = np.array([
        [Wi[j], Wj[j]], 
        [Wi[i], Wi[j]]
    ])
    eigs_B = np.linalg.eig(B)[0]
    max_delta = -1 * (eigs_B[eigs_B < 0])
    if max_delta.shape[0] == 0:
        max_delta = np.inf
    else:
        max_delta = (1/max_delta).max() - 1e-5
    min_delta = -1 * (eigs_B[eigs_B > 0])
    if min_delta.shape[0] == 0:
        min_delta = -np.inf
    else:
        min_delta = (1/min_delta).max() + 1e-5
    
    return objective, min_delta, max_delta

def _solve_cn_cd(Q, R, j):
    """
    Solves for cn/cd using QR decomp. This is 
    specialized for diagonal elements for group knockoffs.
    See https://arxiv.org/abs/2011.14625.
    """
    # Notational note: W = (QR)^{-1}
    Wj = sp.linalg.solve_triangular(a=R, b=Q[j], lower=False)
    cd = Wj[j]
    cn = np.power(Wj, 2).sum()
    return cn, cd

def _solve_mvr_grouped(
    Sigma,
    groups=None,
    tol=1e-5,
    verbose=False,
    num_iter=20,
    converge_tol=1e-3,
    smoothing=0,
    rej_rate=0,
):
    """
    Computes S-matrix for minimum variance-based
    reconstructability group knockoffs.

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
    verbose : bool
        If True, prints updates during optimization.
    num_iter : int
        The number of coordinate descent iterations. Defaults to 50.
    converge_tol : float
        A parameter specifying the criteria for convergence.

    Returns
    -------
    S : np.ndarray
        ``(p, p)``-shaped (block) diagonal matrix used to generate knockoffs
    """

    # Todo
    if smoothing != 0:
        raise NotImplementedError("Smoothing not implemented for group MVR")
    if rej_rate != 0:
        raise NotImplementedError("Rej rate adjustment not implemented for group MVR")

    # group_id maps to group_indices
    block_inds = {}
    for group_id in np.unique(groups):
        block_inds[group_id] = np.where(groups == group_id)[0]

    # Each nonzero element of S
    S_elems = []
    for g_id in block_inds:
        for i_within, i in enumerate(block_inds[g_id]):
            for j_within, j in enumerate(block_inds[g_id]):
                if j > i:
                    break
                S_elems.append(((i, j), (i_within, j_within), g_id))
                
    # Basis elements
    p = Sigma.shape[0]
    basis = []
    for j in range(p):
        ej = np.zeros(p)
        ej[j] = 1
        basis.append(ej)

    # Initial constants
    time0 = time.time()
    loss = np.inf
    acc_rate = 1 - rej_rate

    # Initialize values
    decayed_improvement = 10
    min_eig = np.linalg.eigh(Sigma)[0].min()
    S = mac.solve_equicorrelated(Sigma, groups)/2
    Q, R = np.linalg.qr(2 * Sigma - S + smoothing * np.eye(p))

    # Running QR decompositions of blocks of S
    Sblock_QR = {}
    for g_id in block_inds:
        blocksize = block_inds[g_id].shape[0]
        Sblock = S[block_inds[g_id]][:, block_inds[g_id]]
        # Easy QR decomp for diagonal matrix
        Sblock_QR[g_id] = np.linalg.qr(Sblock)
        
    for it in range(num_iter):
        np.random.shuffle(S_elems)
        # i, j are the coordinates within L
        # i_within, j_within are the coordinates within S
        for (i, j), (i_within, j_within), g_id in S_elems:
            
            # Retrieve QR decomposition for this block of S
            Q_S, R_S = Sblock_QR[g_id]
            # Basis elements within the block
            d = Q_S.shape[0]
            ei = np.zeros(d)
            ei[i_within] = 1
            ej = np.zeros(d)
            ej[j_within] = 1
            
            # For off-diagonal elements: rank-2 update
            if i != j:
               
                # 1. Find optimal delta
                #stime = time.time()
                loss_diff, lb_diff, ub_diff = _mvr_group_contrib(Q, R, i, j)
                max_delta_diff = -1 * lb_diff
                min_delta_diff = -1 * ub_diff
                loss_S, min_delta_S, max_delta_S = _mvr_group_contrib(
                    Q_S, R_S, i=i_within, j=j_within
                )
                lower_bound = np.maximum(min_delta_diff, min_delta_S)
                upper_bound = np.minimum(max_delta_diff, max_delta_S)
                if lower_bound > upper_bound:
                    raise RuntimeError(f"No feasible solutions: lower bound {lower_bound} > upper_bound {upper_bound} in coord descent")
    
                coord_loss = lambda delta: -loss_S(delta) - loss_diff(-1*delta)
                delta = sp.optimize.fminbound(
                    coord_loss, lower_bound, upper_bound
                )
                delta = np.maximum(np.minimum(upper_bound, delta), lower_bound)

                # 2. Update QR decomp for 2 Sigma - S
                U = np.array([basis[i], basis[j]]).T
                V = np.array([basis[j], basis[i]]).T
                Q, R = sp.linalg.qr_update(
                    Q=Q, R=R, u=-1*delta*U, v=V
                )
                # 3. Update QR decomp for S
                U = np.array([ei, ej]).T
                V = np.array([ej, ei]).T
                Sblock_QR[g_id] = sp.linalg.qr_update(
                    Q=Q_S, R=R_S, u=delta*U, v=V
                )

                # 4. Update S
                S[i, j] += delta
                S[j, i] += delta


            # For diagonal elements, rank-1 update
            else:
                cn_diff, cd_diff = _solve_cn_cd(Q, R, j)

                # 2. Compute coefficients cn and cd for S
                cn_S, cd_S = _solve_cn_cd(
                    Q=Q_S,
                    R=R_S,
                    j=j_within
                )

                # 2. Construct/solve quadratic equation
                delta = _solve_mvr_quadratic_group(
                    cn_diff=cn_diff,
                    cd_diff=cd_diff,
                    cn_S=cn_S,
                    cd_S=cd_S,
                )

                # 3. Update QR decmp for 2Sigma - S
                Q, R = sp.linalg.qr_update(
                    Q=Q,
                    R=R,
                    u=basis[i],
                    v=-1*delta*basis[j],
                )

                # 4. Update QR decomp for Sblock
                Sblock_QR[g_id] = sp.linalg.qr_update(
                    Q=Q_S,
                    R=R_S,
                    u=ei,
                    v=delta*ej
                )
                # 5. Set new value for S
                S[i, j] += delta

        # Check for convergence
        prev_loss = loss
        loss = mvr_loss(Sigma, acc_rate * S, smoothing=smoothing)
        if it != 0:
            decayed_improvement = decayed_improvement / 10 + 9 * (prev_loss - loss) / 10
        if verbose:
            print(
                f"After iter {it} at time {np.around(time.time() - time0,3)}, loss={loss}, decayed_improvement={decayed_improvement}"
            )
        if decayed_improvement < converge_tol and decayed_improvement > 0:
            if verbose:
                print(f"Converged after iteration {it} with loss={loss}")
            break

    return S

def solve_mvr(
    Sigma,
    groups=None,
    tol=1e-5,
    verbose=False,
    num_iter=50,
    smoothing=0,
    rej_rate=0,
    converge_tol=1e-3,
    choldate_warning=True,
):
    """
    Computes S-matrix used to generate minimum variance-based
    reconstructability knockoffs using coordinate descent.

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
        If True and groups is None, will warn the user if choldate
        is not installed. Defaults to True.

    Returns
    -------
    S : np.ndarray
        ``(p, p)``-shaped (block) diagonal matrix used to generate knockoffs
    """

    # Process groups
    p = Sigma.shape[0]
    if groups is None:
        groups = np.arange(1, p+1)
    groups = utilities.preprocess_groups(groups)

    # Specialized implementations for ungrouped
    if np.all(groups == np.arange(1, p+1)):
        S = _solve_mvr_ungrouped(
            Sigma=Sigma, 
            tol=tol,
            verbose=verbose,
            num_iter=num_iter,
            smoothing=smoothing,
            rej_rate=rej_rate,
            converge_tol=converge_tol,
            choldate_warning=choldate_warning
        )
    else: 
        S = _solve_mvr_grouped(
            Sigma=Sigma,
            groups=groups,
            tol=tol,
            verbose=verbose,
            num_iter=num_iter,
            converge_tol=converge_tol,
            smoothing=smoothing,
            rej_rate=rej_rate,
        )

    # Ensure validity of solution
    S = utilities.shift_until_PSD(S, tol=tol)
    S, _ = utilities.scale_until_PSD(Sigma, S, tol=tol, num_iter=10)
    return S


### Maxent/MMI
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
        The maxent loss for Sigma and S. This is infinite if S is not feasible.
    """
    p = Sigma.shape[0]
    eigs_S = np.diag(S)
    eigs_diff = np.linalg.eigh(2 * Sigma - S)[0]
    if np.min(eigs_S) < 0 or np.min(eigs_diff) < 0:
        return np.inf

    det_invG = np.log(1/(eigs_diff + smoothing)).sum()
    det_invG = det_invG + np.log(1/(eigs_S + smoothing)).sum()
    return det_invG

def mmi_loss(*args, **kwargs):
    """
    Computes the log determinant of the feature-knockoff precision
    matrix, which is proportional mutual information between X and knockoffs.

    This is identical to ``maxent_loss`` and exists only for backwards 
    compatability.
    """
    return maxent_loss(*args, **kwargs)

def solve_maxent_factored(
    D,
    U,
    tol=1e-5,
    verbose=False,
    num_iter=50,
    converge_tol=1e-4,
):
    """
    Computes S-matrix used to generate maximum entropy
    knockoffs using coordinate descent assuming that
    the covariance matrix follows a factor model.
    This means Sigma = D + UU^T for a p x p diagonal matrix
    D and a p x k matrix U. 

    Parameters
    ----------
    D : np.ndarray
        ``p``-shaped array of diagonal elements.
    U : np.ndarray
        ``(p, k)``-shaped matrix. Usually k << p.
    tol : float
        Minimum permissible eigenvalue of 2Sigma - S and S.
    verbose : bool
        If True, prints updates during optimization.
    num_iter : int
        The number of coordinate descent iterations. Defaults to 50.
    converge_tol : float
        A parameter specifying the criteria for convergence.

    Returns
    -------
    S : np.ndarray
        ``(p, p)``-shaped (block) diagonal matrix used to generate knockoffs
    """
    return _solve_maxent_sdp_factored(
        D=D,
        U=U,
        solve_sdp=False,
        tol=tol,
        verbose=verbose,
        num_iter=num_iter,
        converge_tol=converge_tol
    )


def _solve_maxent_sdp_factored(
    D,
    U,
    solve_sdp,
    tol=1e-5,
    verbose=False,
    num_iter=50,
    converge_tol=1e-4,
    mu=0.9,
    lambd=0.5,
):
    """
    Internal function for maxent/sdp s-matrix using
    factor approximation.

    Parameters
    ----------
    D : np.ndarray
        ``p``-shaped array of diagonal elements.
    U : np.ndarray
        ``(p, k)``-shaped matrix. Usually k << p.
    solve_sdp : bool
        If True, solve for SDP knockoffs. Else, solve for
        maximum entropy knockoffs.
    tol : float
        Minimum permissible eigenvalue of 2Sigma - S and S.
    verbose : bool
        If True, prints updates during optimization.
    num_iter : int
        The number of coordinate descent iterations. Defaults to 50.
    converge_tol : float
        A parameter specifying the criteria for convergence.
    mu : float
        Barrier decay parameter
    lambd : float
        Initial barrier parameter

    Returns
    -------
    S : np.ndarray
        ``(p, p)``-shaped (block) diagonal matrix used to generate knockoffs
    """

    # Initial constants
    p = D.shape[0]
    k = U.shape[1]
    inds = np.arange(p)
    loss = np.inf

    # TODO: this is Omega(p^3).
    # Maybe use https://www.sciencedirect.com/science/article/pii/S2215016118300062
    Sigma = np.diag(D) + np.dot(U, U.T)
    diag_Sigma = np.diag(Sigma)
    mineig = np.linalg.eigh(Sigma)[0].min()
    if mineig < 0:
        raise ValueError("D + UU^T is not PSD")

    # Initialize values
    time0 = time.time()
    decayed_improvement = 1
    if solve_sdp:
        Sdiag = np.zeros(p) + 0.01 * mineig
    else:
        Sdiag = np.zeros(p) + mineig
    lambd = min(2*mineig, lambd)

    # These are all k x k matrices
    Q, R = sp.linalg.qr(np.eye(k) + 2*np.dot(U.T / (2*D - Sdiag), U))

    for i in range(num_iter):
        np.random.shuffle(inds)
        for j in inds:
            # Qprime, Rprime are rank-one updates of Q, R
            # to help solve for the optimal Sj
            Qprime, Rprime = sp.linalg.qr_update(
                Q=Q,
                R=R,
                u=U[j].T/(2*D[j] - Sdiag[j]),
                v=-2*U[j].T,
                overwrite_qruv=False
            )

            # Calculate update for S---for notation, see
            # https://arxiv.org/pdf/2006.08790.pdf
            MjUjT = np.dot(np.dot(Qprime, Rprime) - np.eye(k), U[j].T)/2
            x = sp.linalg.solve_triangular(
                a=Rprime,
                b=np.dot(Qprime.T, MjUjT),
                lower=False
            )
            sub_term = 4*np.dot(U[j], MjUjT) - 8*np.dot(MjUjT, x)
            if solve_sdp:
                Sjstar = max(min(1, 2*diag_Sigma[j] - sub_term - lambd), 0)
            else:
                Sjstar = (2*diag_Sigma[j] - sub_term)/2
            
            # Rank one update to QR decomp
            delta = Sdiag[j] - Sjstar 
            muj = U[j].T / (2*D[j] - Sdiag[j])
            denom = 1 + delta/(2*D[j] - Sdiag[j])
            c = delta/denom
            Q, R = sp.linalg.qr_update(
                Q=Q,
                R=R,
                u=-2*c*muj,
                v=muj,
            )

            # Update S
            Sdiag[j] = Sjstar

        if solve_sdp:
            lambd = mu * lambd

    return np.diag(Sdiag)


def solve_maxent(
    Sigma, 
    tol=1e-5,
    verbose=False,
    num_iter=50,
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
    return _solve_maxent_sdp_cd(
        Sigma=Sigma,
        tol=tol,
        verbose=verbose,
        num_iter=num_iter,
        converge_tol=converge_tol,
        choldate_warning=choldate_warning,
        solve_sdp=False,
    )


def _solve_maxent_sdp_cd(
    Sigma, 
    solve_sdp,
    tol=1e-5,
    verbose=False,
    num_iter=50,
    converge_tol=1e-4,
    choldate_warning=True,
    mu=0.9,
    lambd=0.5,
):
    """
    This function is internally used to compute the S-matrices
    used to generate maximum entropy and SDP knockoffs. Users
    should not call this function---they should call ``solve_maxent``
    or ``solve_sdp`` directly.

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
    solve_sdp : bool
        If True, will solve SDP. Otherwise, will solve maxent formulation.
    lambd : float
        Initial barrier constant
    mu : float
        Barrier decay constant

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
    mineig = np.linalg.eigh(V)[0].min()
    if solve_sdp:
        S = 0.01 * mineig * np.eye(p)
    else:
        S = mineig * np.eye(p)
    L = np.linalg.cholesky(2 * V - S)
    lambd = min(2*mineig, lambd)

    # Loss function
    if solve_sdp:
        loss_fn = lambda V, S: S.shape[0] - np.diag(S).sum()
    else:
        loss_fn = maxent_loss

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
            if solve_sdp:
                sjstar = max(min(1, 2 * V[j, j] - qinvterm - lambd), 0)
            else:
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
        loss = loss_fn(V, S)
        if i != 0:
            loss_diff = prev_loss - loss 
            if solve_sdp:
                loss_diff = max(loss_diff, lambd)
            decayed_improvement = decayed_improvement / 10 + 9 * (loss_diff) / 10
        if verbose:
            print(
                f"After iter {i} at time {np.around(time.time() - time0,3)}, loss={loss}, decayed_improvement={decayed_improvement}"
            )
        if decayed_improvement < converge_tol:
            if verbose:
                print(f"Converged after iteration {i} with loss={loss}")
            break

        # Update barrier parameter if solving SDP
        if solve_sdp:
            lambd = mu * lambd

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
