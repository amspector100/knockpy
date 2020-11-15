import warnings
import numpy as np
import scipy as sp
import choldate
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utilities
from .utilities import calc_group_sizes, preprocess_groups

# Multiprocessing tools
import itertools
import time
from functools import partial
from multiprocessing import Pool

DEFAULT_TOL = 1e-5
GAMMA_VALS = [
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
    0.9999,
    1
]

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
### Gradient Based Solver

def blockdiag_to_blocks(M, groups):
    """
    Given a matrix M, pulls out the diagonal blocks as specified by groups.
    :param M: p x p numpy array
    :param groups: p length numpy array 
    """
    blocks = []
    for j in np.sort(np.unique(groups)):
        inds = np.where(groups == j)[0]
        full_inds = np.ix_(inds, inds)
        blocks.append(M[full_inds].copy())
    return blocks

def blocks_to_blockdiag(blocks, block_inds):
    """
    Given a list of square matrices, arranges them into a permuted block-diagonal
    matrix according to block_inds.
    :param blocks: A list of m block matrices.
    :param block_inds: A p-length numpy array with values from 1 to
    returns: p x p block-diagonal numpy array. 
    """

def block_diag_sparse(*arrs):
    """ Given a list of 2D torch tensors, creates a sparse block-diagonal matrix
    See https://github.com/pytorch/pytorch/issues/31942
    """
    bad_args = []
    for k, arr in enumerate(arrs):
        # if isinstance(arr, nn.Parameter):
        #     arr = arr.data
        if not (isinstance(arr, torch.Tensor) and arr.ndim == 2):
            bad_args.append(k)

    if len(bad_args) != 0:
        raise ValueError(f"Args in {bad_args} positions must be 2D tensor")

    shapes = torch.tensor([a.shape for a in arrs])
    out = torch.zeros(
        torch.sum(shapes, dim=0).tolist(), dtype=arrs[0].dtype, device=arrs[0].device
    )
    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r : r + rr, c : c + cc] = arrs[i]
        r += rr
        c += cc
    return out

class MVRLoss(nn.Module):
    """
    A pytorch class to compute S-matrices for 
    (gaussian) MX knockoffs which minimizes the 
    trace of the feature-knockoff precision matrix
    (the inverse of the feature-knockoff 
    covariance/Grahm matrix, G).

    :param Sigma: p x p numpy matrix. Must already
    be sorted by groups.
    :param groups: p length numpy array of groups. 
    These must already be sorted and correspond to 
    the Sigma.
    :param init_S: The initialization values for the S
    block-diagonal matrix. 
    - A p x p matrix. The block-diagonal of this matrix,
    as specified by groups, will be the initial values 
    for the S matrix.
    - A list of square numpy matrices, with the ith element
    corresponding to the block of the ith group in S.
    Default: Half of the identity.
    :param rec_prop: The proportion of data you are planning
    to recycle. (The optimal S matrix depends on the recycling
    proportion.)
    :param rec_prop: The proportion of knockoffs that will be
    recycled.
    :param smoothing: Calculate the loss as sum 1/(eigs + smoothing)
    as opposed to sum 1/eigs. This is helpful if fitting lasso 
    statistics on extremely degenerate covariance matrices. Over the 
    course of optimization, this smoothing parameter will go to 0.
    :param method: One of mvr or mmi.
    """

    def __init__(self, 
        Sigma,
        groups,
        init_S=None,
        invSigma=None,
        rec_prop=0,
        smoothing=0.01,
        min_smoothing=1e-4,
        method='mvr',
    ):

        super().__init__()

        # Groups MUST be sorted
        sorted_groups = np.sort(groups)
        if not np.all(groups == sorted_groups):
            raise ValueError("Sigma and groups must be sorted prior to input")

        # Save sigma and groups
        self.p = Sigma.shape[0]
        self.groups = torch.from_numpy(groups).long()
        self.group_sizes = torch.from_numpy(calc_group_sizes(groups)).long()
        self.Sigma = torch.from_numpy(Sigma).float()

        # Save inverse cov matrix, rec_prop
        if invSigma is None:
            invSigma = utilities.chol2inv(Sigma)
        self.invSigma = torch.from_numpy(invSigma).float()

        # Save recycling proportion and smoothing
        self.smoothing = smoothing
        self.min_smoothing = min_smoothing
        self.rec_prop = rec_prop
        self.method = method

        # Make sure init_S is a numpy array
        if init_S is None:
            # If nothing provided, default to equicorrelated
            scale = min(1, 2 * np.linalg.eigh(Sigma)[0].min())
            init_S = scale * np.eye(self.p)
        elif isinstance(init_S, list):
            # Check for correct number of blocks
            num_blocks = len(init_S)
            num_groups = np.unique(groups).shape[0]
            if num_blocks != num_groups:
                raise ValueError(
                    f"Length of init_S {num_blocks} doesn't agree with num groups {num_groups}"
                )
            init_S = sp.linalg.block_diag(*init_S)

        # Find a good initial scaling
        best_gamma = 1
        best_loss = np.inf
        for gamma in GAMMA_VALS:
            loss = mvr_loss(
                Sigma=Sigma,
                S=(1-self.rec_prop)*gamma*init_S,
            )
            if loss >= 0 and loss < best_loss:
                best_gamma = gamma
                best_loss = loss
        init_S = best_gamma * init_S

        # Create new blocks
        blocks = blockdiag_to_blocks(init_S, groups)
        # Torch-ify and take sqrt
        blocks = [torch.from_numpy(block) for block in blocks]
        blocks = [torch.cholesky(block) for block in blocks]
        # Save
        self.blocks = [nn.Parameter(block.float()) for block in blocks]

        # Register the blocks as parameters
        for i, block in enumerate(self.blocks):
            self.register_parameter(f"block{i}", block)

        self.update_sqrt_S()
        self.scale_sqrt_S(tol=1e-5, num_iter=10)

    def update_sqrt_S(self):
        """ Updates sqrt_S using the block parameters """
        self.sqrt_S = block_diag_sparse(*self.blocks)

    def pull_S(self):
        """ Returns the S matrix """
        self.update_sqrt_S()
        S = torch.mm(self.sqrt_S.t(), self.sqrt_S)
        return S

    def forward(self, smoothing = None):
        """ Calculates trace of inverse grahm feature-knockoff matrix"""

        # TODO: This certainly works and is more efficient in a forward
        # pass than taking the eigenvalues of both S and 2*Sigma - S.
        # But perhaps the dot product makes the backprop less efficient?

        # Infer smoothing
        if smoothing is None:
            smoothing = self.smoothing

        # Create schurr complement
        S = self.pull_S()
        S = (1 - self.rec_prop) * S  # Account for recycling calcing loss
        diff = self.Sigma - S
        G_schurr = self.Sigma - torch.mm(torch.mm(diff, self.invSigma), diff)

        # Take eigenvalues
        eigvals = torch.symeig(G_schurr, eigenvectors=True)
        eigvals = eigvals[0]
        if self.method == 'mvr':
            inv_eigs = 1 / (smoothing + eigvals)
        elif self.method == 'mmi':
            inv_eigs = torch.log(
                1 / torch.max((smoothing + eigvals), torch.tensor(smoothing).float()),
            )
        return inv_eigs.sum()


    def scale_sqrt_S(self, tol, num_iter):
        """ Scales sqrt_S such that 2 Sigma - S is PSD."""

        # No gradients
        with torch.no_grad():

            # This shift only applies for 
            for block in self.blocks:
                if block.data.shape[0] == 1:
                    block.data = torch.max(torch.tensor(tol).float(), block.data)


            # Construct S
            S = self.pull_S()
            # Find optimal scaling
            _, gamma = utilities.scale_until_PSD(
                self.Sigma.numpy(), S.numpy(), tol=tol, num_iter=num_iter
            )
            # Scale blocks
            for block in self.blocks:
                block.data = np.sqrt(gamma) * block.data
            self.update_sqrt_S()

    def project(self, **kwargs):
        """ Project by scaling sqrt_S """
        self.scale_sqrt_S(**kwargs)


class PSGDSolver:
    """ 
    Projected gradient descent to solve for MRC knockoffs.
    This will work for non-convex loss objectives as well,
    although it's a heuristic optimization method.
    :param Sigma: p x p numpy array, the correlation matrix
    :param groups: p-length numpy array specifying groups
    :param losscalc: A pytorch class wrapping nn.module
    which contains the following methods:
    - .forward() which calculates the loss based on the
    internally stored S matrix.
    - .project() which ensures that both the internally-stored
    S matrix as well as (2*Sigma - S) are PSD.
    - .pull_S(), which returns the internally-stored S matrix.
    If None, creates a MVRLoss class. 
    :param lr: Initial learning rate (default 1e-2)
    :param verbose: if true, reports progress
    :param max_epochs: Maximum number of epochs in SGD
    :param tol: Mimimum eigenvalue allowed for PSD matrices
    :param line_search_iter: Number of line searches to do
    when scaling sqrt_S.
    :param convergence_tol: After each projection, we calculate
    improvement = 2/3 * ||prev_opt_S - opt_S||_1 + 1/3 * (improvement)
    When improvement < convergence_tol, we return.
    :param kwargs: Passed to MVRLoss 
    """

    def __init__(
        self,
        Sigma,
        groups,
        losscalc=None,
        lr=1e-2,
        verbose=False,
        max_epochs=100,
        tol=1e-5,
        line_search_iter=10,
        convergence_tol=1e-1,
        **kwargs
    ):

        # Add Sigma
        self.p = Sigma.shape[0]
        self.Sigma = Sigma
        self.groups = groups
        self.opt_S = None # Output initialization
        self.opt_loss = np.inf

        # Save parameters for optimization
        self.lr = lr
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.tol = tol
        self.line_search_iter = line_search_iter
        self.convergence_tol = convergence_tol

        # Sort by groups for ease of computation
        inds, inv_inds = utilities.permute_matrix_by_groups(groups)
        self.inds = inds
        self.inv_inds = inv_inds
        self.sorted_Sigma = self.Sigma[inds][:, inds]
        self.sorted_groups = self.groups[inds]

        # Loss calculator
        if losscalc is not None:
            self.losscalc = losscalc
        else:
            self.losscalc = MVRLoss(
                Sigma=self.sorted_Sigma, groups=self.sorted_groups, **kwargs
            )

        # Initialize cache of optimal S
        with torch.no_grad():
            init_loss = self.losscalc(smoothing=0)
            if init_loss < 0:
                init_loss = np.inf
        self.cache_S(init_loss)

        # Initialize attributes which save losses over time
        self.all_losses = []
        self.projected_losses = []
        self.improvements = []

    def cache_S(self, new_loss):
        # Cache optimal solution
        with torch.no_grad():
            self.prev_opt_S = self.opt_S
            self.prev_opt_loss = self.opt_loss
            self.opt_loss = new_loss
            self.opt_S = self.losscalc.pull_S().clone().detach().numpy()

    def optimize(self):
        """
        See __init__ for arguments.
        """
        # Optimizer
        params = list(self.losscalc.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        improvement = self.convergence_tol + 10
        for j in range(self.max_epochs):

            # Step 1: Calculate loss (trace of feature-knockoff precision)
            loss = self.losscalc()
            if np.isnan(loss.detach().item()):
                warnings.warn(
                    f"Loss of {self.losscalc.method} solver is NaN"
                )
                break
            self.all_losses.append(loss.item())

            # Step 2: Step along the graient
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()


            # Step 3: Reproject to be PSD
            if j % 3 == 0 or j == self.max_epochs - 1:
                self.losscalc.project(tol=self.tol, num_iter=self.line_search_iter)

                # If this is optimal after reprojecting, save
                with torch.no_grad():
                    new_loss = self.losscalc(smoothing = 0)
                if new_loss < self.opt_loss and new_loss >= 0:
                    self.cache_S(new_loss)
                else:
                    self.prev_opt_S = self.opt_S
                    self.prev_opt_loss = self.opt_loss

                # Cache projected loss
                self.projected_losses.append(new_loss.item())

                # Calculate improvement
                if j != 0 and j % 10 == 0:
                    diff = self.prev_opt_loss - self.opt_loss
                    l1diff = np.abs(self.opt_S - self.prev_opt_S).sum()
                    improvement = 2*(diff)/3 + improvement/3
                    if self.verbose:
                        print(f"L1 diff is {l1diff}, loss diff={diff}, improvement is {improvement}, best loss is {self.opt_loss} at iter {j}")
                self.improvements.append(improvement)

                # Break if improvement is small
                if improvement < self.convergence_tol and j % 10 == 0:
                    if self.losscalc.smoothing > self.losscalc.min_smoothing:
                        improvement = 1 + convergence_tol # Reset
                        self.losscalc.smoothing = max(self.losscalc.min_smoothing, self.losscalc.smoothing / 10)
                        if self.verbose:
                            print(f"Nearing convergence, reducing smoothing to {self.losscalc.smoothing} \n")
                    elif self.verbose:
                        print(f"Converged at iteration {j}")
                    break

        # Shift, scale, and return
        sorted_S = self.opt_S
        S = sorted_S[self.inv_inds][:, self.inv_inds]
        S = utilities.shift_until_PSD(S, tol=self.tol)
        S, _ = utilities.scale_until_PSD(
            self.Sigma, S, tol=self.tol, num_iter=self.line_search_iter
        )
        return S

def solve_mrc_psgd(
    Sigma,
    groups=None,
    method='mvr',
    **kwargs,
):
    """
    Wraps the PSGDSolver class.
    :param Sigma: Covariance matrix
    :param groups: groups for group knockoffs
    :param method: MRC loss (mvr or mmi)
    :param init_kwargs: kwargs to pass to 
    PSGDSolver.
    :param optimize_kwargs: kwargs to pass 
    to optimizer method.
    :returns: opt_S
    """
    solver = PSGDSolver(
        Sigma=Sigma,
        groups=groups,
        method=method,
        **kwargs
    )
    opt_S = solver.optimize()
    return opt_S