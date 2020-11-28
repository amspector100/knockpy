""" Gradient-based methods for solving MRC problems.
Currently only used for group-knockoffs."""

import warnings
import time
import numpy as np
import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import GAMMA_VALS
from .. import utilities, mrc


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
    :param method: One of mvr or maxent (mmi for backwards compatability).
    """

    def __init__(
        self,
        Sigma,
        groups,
        init_S=None,
        invSigma=None,
        rec_prop=0,
        smoothing=0.01,
        min_smoothing=1e-4,
        method="mvr",
    ):

        super().__init__()

        # Groups MUST be sorted
        sorted_groups = np.sort(groups)
        if not np.all(groups == sorted_groups):
            raise ValueError("Sigma and groups must be sorted prior to input")

        # Save sigma and groups
        self.p = Sigma.shape[0]
        self.groups = torch.from_numpy(groups).long()
        self.group_sizes = torch.from_numpy(utilities.calc_group_sizes(groups)).long()
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
        if method == "mvr":
            objective = mrc.mvr_loss
        else:
            objective = mrc.maxent_loss
        for gamma in GAMMA_VALS:
            loss = objective(Sigma=Sigma, S=(1 - self.rec_prop) * gamma * init_S,)
            if loss >= 0 and loss < best_loss:
                best_gamma = gamma
                best_loss = loss
        init_S = best_gamma * init_S

        # Create new blocks
        blocks = utilities.blockdiag_to_blocks(init_S, groups)
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

    def forward(self, smoothing=None):
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
        if self.method == "mvr":
            inv_eigs = 1 / (smoothing + eigvals)
        elif self.method == "maxent":
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
        **kwargs,
    ):

        # Add Sigma
        self.p = Sigma.shape[0]
        self.Sigma = Sigma
        self.groups = groups
        self.opt_S = None  # Output initialization
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
                warnings.warn(f"Loss of {self.losscalc.method} solver is NaN")
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
                    new_loss = self.losscalc(smoothing=0)
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
                    improvement = 2 * (diff) / 3 + improvement / 3
                    if self.verbose:
                        print(
                            f"L1 diff is {l1diff}, loss diff={diff}, improvement is {improvement}, best loss is {self.opt_loss} at iter {j}"
                        )
                self.improvements.append(improvement)

                # Break if improvement is small
                if improvement < self.convergence_tol and j % 10 == 0:
                    if self.losscalc.smoothing > self.losscalc.min_smoothing:
                        improvement = 1 + convergence_tol  # Reset
                        self.losscalc.smoothing = max(
                            self.losscalc.min_smoothing, self.losscalc.smoothing / 10
                        )
                        if self.verbose:
                            print(
                                f"Nearing convergence, reducing smoothing to {self.losscalc.smoothing} \n"
                            )
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
    Sigma, groups=None, method="mvr", **kwargs,
):
    """
    Wraps the PSGDSolver class.
    :param Sigma: Covariance matrix
    :param groups: groups for group knockoffs
    :param method: MRC loss (mvr or maxent)
    :param init_kwargs: kwargs to pass to 
    PSGDSolver.
    :param optimize_kwargs: kwargs to pass 
    to optimizer method.
    :returns: opt_S
    """
    solver = PSGDSolver(Sigma=Sigma, groups=groups, method=method, **kwargs)
    opt_S = solver.optimize()
    return opt_S
