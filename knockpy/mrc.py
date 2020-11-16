import warnings
import time
import numpy as np
import scipy as sp
import choldate
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utilities
from .utilities import calc_group_sizes, preprocess_groups

### Coordinate Descent Solvers
def mvr_loss(Sigma, S, smoothing=0):
    """ Computes trace of inverse of feature-knockoff
    precision matrix using numpy (no backprop) """

    # Inverse of eigenvalues
    trace_invG = (1 / (np.linalg.eigh(2*Sigma - S)[0] + smoothing)).sum()
    trace_invG = trace_invG + (1 / (np.diag(S) + smoothing)).sum()
    return trace_invG

def mmi_loss(Sigma, S, smoothing=0):
    """
    Computes (smoothed) determinant of feature-knockoff covariance.
    Does not support group knockoffs as of yet.
    """
    p = Sigma.shape[0]
    detG = np.log(np.linalg.det(2*Sigma - S + smoothing * np.eye(p)))
    detG = detG + np.log(np.diag(S+smoothing)).sum()
    return -1*detG

def solve_mvr(
    Sigma,
    tol=1e-5,
    verbose=False, 
    num_iter=10,
    smoothing=0, 
    rej_rate=0,
    converge_tol=1
):
    """
    Uses a coordinate-descent algorithm to find the solution to the smoothed
    MVR problem. 
    :param Sigma: p x p covariance matrix
    :param tol: Minimum eigenvalue of 2Sigma - S and S
    :param num_iter: Number of coordinate descent iterations
    :param rej_rate: Expected proportion of rejections for knockoffs under the
    metropolized knockoff sampling framework.
    :param verbose: if true, will give progress reports
    :param smoothing: computes smoothed mvr loss
    """

    # Initial constants
    time0 = time.time()
    V = Sigma # I'm too lazy to write Sigma out over and over
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
    L = np.linalg.cholesky(2*V - S + smoothing * np.eye(p))

    for i in range(num_iter):
        np.random.shuffle(inds)
        for j in inds:
            # 1. Compute coefficients cn and cd
            ej = np.zeros(p) # jth basis element
            ej[j] = 1
            # 1a. Compute cd 
            vd = sp.linalg.solve_triangular(a=L, b=ej, lower=True)
            cd = np.power(vd, 2).sum()
            # 1b. Compute vn
            vn = sp.linalg.solve_triangular(a=L.T, b=vd, lower=False)
            cn = -1*np.power(vn, 2).sum()

            # 2. Construct quadratic equation
            # We want to minimize 1/(sj + delta) - (delta * cn)/(1 - delta * cd)
            coef2 = -1*cn - np.power(cd, 2)
            coef1 = 2*(-1*cn*(S[j,j]+smoothing) + cd)
            coef0 = -1*cn*(S[j,j]+smoothing)**2 - 1
            orig_options = np.roots(np.array([coef2,coef1,coef0]))

            # 3. Eliminate complex solutions
            options = np.array([
                delta for delta in orig_options if np.imag(delta)==0
            ])
            # Eliminate solutions which violate PSD-ness
            upper_bound = 1 / cd
            lower_bound = -1*S[j,j]
            options = np.array([
                delta for delta in options if delta < upper_bound and delta > lower_bound
            ])
            if options.shape[0] == 0:
                raise RuntimeError(f"All quadratic solutions ({orig_options}) were infeasible or imaginary")

            # 4. If multiple solutions left (unlikely), pick the smaller one
            losses = 1/(S[j,j] + options) - (options * cn)/(1 - options*cd)
            if losses[0] == losses.min():
                delta = options[0]
            else:
                delta = options[1]
            
            # 5. Account for rejections
            if acc_rate < 1:
                extra_space = min(min_eig, 0.02)/(i+2) # Helps deal with coord desc
                opt_postrej_value = S[j,j] + delta
                opt_prerej_value = opt_postrej_value / (acc_rate)
                opt_prerej_value = min(
                    S[j,j]+upper_bound-extra_space,
                    max(opt_prerej_value, extra_space)
                )
                delta = opt_prerej_value - S[j,j]

            # Update S and L
            x = np.zeros(p)
            x[j] = np.sqrt(np.abs(delta))
            if delta > 0:
                choldate.choldowndate(L.T, x)
            else:
                choldate.cholupdate(L.T, x)
            
            # Set new value for S
            S[j,j] += delta

        # Check for convergence
        prev_loss = loss
        loss = mvr_loss(V, acc_rate*S, smoothing=smoothing)
        if i != 0:
            decayed_improvement = decayed_improvement / 10 + 9*(prev_loss - loss) / 10
        if verbose:
                print(f"After iter {i} at time {np.around(time.time() - time0,3)}, loss={loss}, decayed_improvement={decayed_improvement}")
        if decayed_improvement < converge_tol:
            if verbose:
                print(f"Converged after iteration {i} with loss={loss}")
            break
    # Ensure validity of solution
    S = utilities.shift_until_PSD(S, tol=tol)
    S, _ = utilities.scale_until_PSD(
        V, S, tol=tol, num_iter=10
    )
    return S

def solve_ciknock(
    Sigma, 
    tol=1e-5,
    num_iter=10,
):
    # Compute vanilla S_CI 
    S = 1/(np.diag(np.linalg.inv(Sigma)))
    S = np.diag(S)
    # Ensure validity of solution
    S = utilities.shift_until_PSD(S, tol=tol)
    S, _ = utilities.scale_until_PSD(
        Sigma, S, tol=tol, num_iter=num_iter
    )
    return S

def solve_mmi(
    Sigma,
    tol=1e-5,
    verbose=False, 
    num_iter=10,
    smoothing=0, 
    converge_tol=1e-4
):
    """
    Uses a coordinate-descent algorithm to find the solution to the
    minimum mutual information. 
    :param Sigma: p x p covariance matrix
    :param tol: Minimum eigenvalue of 2Sigma - S and S
    :param num_iter: Number of coordinate descent iterations
    :param verbose: if true, will give progress reports
    :param smoothing: computes smoothed mmi loss
    """

    if smoothing > 0:
        raise NotImplementedError(f"Smoothing is not implemented yet")

    # Initial constants
    time0 = time.time()
    V = Sigma # I'm too lazy to write Sigma out over and over
    p = V.shape[0]
    inds = np.arange(p)
    loss = np.inf

    # Initialize values
    decayed_improvement = 1
    S = np.linalg.eigh(V)[0].min() * np.eye(p)
    L = np.linalg.cholesky(2*V - S)

    for i in range(num_iter):
        np.random.shuffle(inds)
        for j in inds:
            diff = 2*V - S
            
            # Solve cholesky equation
            tildey = 2*V[j].copy()
            tildey[j] = 0
            x = sp.linalg.solve_triangular(a=L, b=tildey, lower=True)
            
            # Use cholesky eq to get new update
            zeta = diff[j,j]
            x22 = np.power(x, 2).sum()
            qinvterm = zeta * x22 / (zeta + x22)
            
            # Inverse of Qj using SWM formula
            sjstar = (2*V[j, j] - qinvterm)/2
            
            # Rank one update for cholesky
            delta = S[j,j] - sjstar
            x = np.zeros(p)
            x[j] = np.sqrt(np.abs(delta))
            if delta > 0:
                choldate.cholupdate(L.T, x)
            else:
                choldate.choldowndate(L.T, x)
            
            # Set new value for S
            S[j,j] = sjstar

        # Check for convergence
        prev_loss = loss
        loss = mmi_loss(V, S, smoothing=smoothing)
        if i != 0:
            decayed_improvement = decayed_improvement / 10 + 9*(prev_loss - loss) / 10
        if verbose:
                print(f"After iter {i} at time {np.around(time.time() - time0,3)}, loss={loss}, decayed_improvement={decayed_improvement}")
        if decayed_improvement < converge_tol:
            if verbose:
                print(f"Converged after iteration {i} with loss={loss}")
            break

    # Ensure validity of solution
    S = utilities.shift_until_PSD(S, tol=tol)
    S, _ = utilities.scale_until_PSD(
        V, S, tol=tol, num_iter=10
    )
    return S