import warnings
import numpy as np
import scipy as sp
from scipy import stats
import scipy.linalg

from .utilities import calc_group_sizes, preprocess_groups
from .utilities import shift_until_PSD, scale_until_PSD
from . import utilities, smatrix


### Base Knockoff Class and Gaussian Knockoffs


class KnockoffSampler:
    """ Base class for sampling knockoffs."""

    def __init__(self):

        pass

    def sample_knockoffs(self):

        raise NotImplementedError()

    def fetch_S(self):
        """ Fetches knockoff S-matrix."""

        raise NotImplementedError

    def check_PSD_condition(self, Sigma, S):
        """ Checks that the feature-knockoff cov matrix is PSD.

        Parameters
        ----------
        Sigma : np.ndarray
            ``(p, p)``-shaped covariance matrix of the features. If None, this
            is estimated using the ``shrinkage`` option. This is ignored for
            fixed-X knockoffs.
        S : np.ndarray
            the ``(p, p)``-shaped knockoff S-matrix used to generate knockoffs. 

        Raises
        ------
        Raises an error if S is not PSD or 2 Sigma - S is not PSD.          
        """

        # Check PSD condition
        min_eig1 = np.linalg.eigh(2 * Sigma - S)[0].min()
        if self.verbose:
            print(f"Minimum eigenvalue of S is {np.linalg.eigh(S)[0].min()}")
            print(f"Minimum eigenvalue of 2Sigma - S is {min_eig1}")
        if min_eig1 < -1e-6:
            raise np.linalg.LinAlgError(
                f"Minimum eigenvalue of 2Sigma - S is {min_eig1}, meaning FDR control violations are extremely likely"
            )

    def many_ks_tests(self, sample1s, sample2s):
        """
        Samples1s, Sample2s = list of arrays
        Gets p values by running ks tests and then
        does a multiple testing correction.
        """
        # KS tests
        pvals = []
        for s, sk in zip(sample1s, sample2s):
            result = stats.ks_2samp(s, sk)
            pvals.append(result.pvalue)
        pvals = np.array(pvals)

        # Naive Bonferroni correction
        adj_pvals = np.minimum(pvals.shape[0]*pvals, 1)
        return pvals, adj_pvals

    def check_xk_validity(self, X, Xk, testname="", alpha=0.001):
        """
        Runs a variety of KS tests on X and Xk to (informally)
        check that Xk are valid knockoffs for X.  Uses the BHQ
        adjustment for multiple testing.

        Parameters
        ----------
        X : np.ndarray
            the ``(n, p)``-shaped design
        Xk : np.ndarray
            the ``(n, p)``-shaped matrix of knockoffs
        testname : str
            a testname that shows up in the error
        alpha : float
            The significance level. Defaults to 0.001
        """
        n = X.shape[0]
        p = X.shape[1]

        # Marginal KS tests
        marg_pvals, marg_adj_pvals = self.many_ks_tests(
            sample1s=[X[:, j] for j in range(p)], sample2s=[Xk[:, j] for j in range(p)]
        )
        min_adj_pval = marg_adj_pvals.min()
        if min_adj_pval < alpha:
            raise ValueError(
                f"For testname={testname}, MARGINAL ks tests reject with min_adj_pval={min_adj_pval}"
            )

        # Pairwise KS tests
        pair_pvals, pair_adj_pvals = self.many_ks_tests(
            sample1s=[X[:, j] * X[:, j + 1] for j in range(p - 1)],
            sample2s=[Xk[:, j] * Xk[:, j + 1] for j in range(p - 1)],
        )
        min_adj_pval = pair_adj_pvals.min()
        if min_adj_pval < alpha:
            raise ValueError(
                f"For testname={testname}, PAIRED ks tests reject with min_adj_pval={min_adj_pval}"
            )

        # Pair-swapped KS tests
        pswap_pvals, pswap_adj_pvals = self.many_ks_tests(
            sample1s=[X[:, j] * Xk[:, j + 1] for j in range(p - 1)],
            sample2s=[Xk[:, j] * X[:, j + 1] for j in range(p - 1)],
        )
        min_adj_pval = pswap_adj_pvals.min()
        if min_adj_pval < alpha:
            raise ValueError(
                f"For testname={testname}, PAIR SWAPPED ks tests reject with min_adj_pval={min_adj_pval}"
            )


def produce_MX_gaussian_knockoffs(X, mu, invSigma, S, sample_tol, copies, verbose):

    # Calculate MX knockoff moments...
    n, p = X.shape
    invSigma_S = np.dot(invSigma, S)
    mu_k = X - np.dot(X - mu.reshape(1, -1), invSigma_S)  # This is a bottleneck??
    Vk = 2 * S - np.dot(S, invSigma_S)

    # Account for numerical errors
    min_eig = np.linalg.eigh(Vk)[0].min()
    if min_eig < sample_tol and verbose:
        warnings.warn(
            f"Minimum eigenvalue of Vk is {min_eig}, under tolerance {sample_tol}"
        )
        Vk = shift_until_PSD(Vk, sample_tol)

    # ...and sample MX knockoffs!
    knockoffs = stats.multivariate_normal.rvs(mean=np.zeros(p), cov=Vk, size=copies * n)

    # Account for case when n * copies == 1
    if n * copies == 1:
        knockoffs = knockoffs.reshape(-1, 1)

    # (Save this for testing later)
    first_row = knockoffs[0, 0:n].copy()

    # Some annoying reshaping...
    knockoffs = knockoffs.flatten(order="C")
    knockoffs = knockoffs.reshape(p, n, copies, order="F")
    knockoffs = np.transpose(knockoffs, [1, 0, 2])

    # (Test we have reshaped correctly)
    new_first_row = knockoffs[0, 0:n, 0]
    np.testing.assert_array_almost_equal(
        first_row,
        new_first_row,
        err_msg="Critical error - reshaping failed in knockoff generator",
    )

    # Add mu
    mu_k = np.expand_dims(mu_k, axis=2)
    knockoffs = knockoffs + mu_k
    return knockoffs


class GaussianSampler(KnockoffSampler):
    """ 
    Samples MX Gaussian (group) knockoffs.

    Parameters
    ----------

    X : np.ndarray
        the ``(n, p)``-shaped design
    mu : np.ndarray
        ``(p, )``-shaped mean of the features. If None, this defaults to
        the empirical mean of the features.
    Sigma : np.ndarray
        ``(p, p)``-shaped covariance matrix of the features. If None, this
        is estimated using the ``utilities.estimate_covariance`` function.
    groups : np.ndarray
                For group knockoffs, a p-length array of integers from 1 to 
                num_groups such that ``groups[j] == i`` indicates that variable `j`
                is a member of group `i`. Defaults to None (regular knockoffs). 
    S : np.ndarray
        the ``(p, p)``-shaped knockoff S-matrix used to generate knockoffs. This 
        is defined such that Cov(X, tilde(X)) = Sigma - S. When None,
        will be constructed by knockoff generator. Defaults to None.
    method : str
        Specifies how to construct S matrix. This will be ignored if ``S`` is not None.
        There are several options:
            - 'mvr': Minimum Variance-Based Reconstructability knockoffs.
            - 'mmi': Minimizes the mutual information between X and the knockoffs.
            - 'ci': Conditional independence knockoffs.
            - 'sdp': minimize the mean absolute covariance (MAC) between the features
            and the knockoffs.
            - 'equicorrelated': Minimizes the MAC under the constraint that the 
            the correlation between each feature and its knockoff is the same.
        The default is to use mvr for non-group knockoffs, and to use the group-SDP
        for grouped knockoffs (the implementation for group mvr knockoffs is currently
        fairly slow). In both cases we use a block-diagonal approximation 
        if the number if features is greater than 1000.
    objective : str
        How to optimize the S matrix if using the SDP for group knockoffs.
        There are several options:
            - 'abs': minimize sum(abs(Sigma - S))
            between groups and the group knockoffs.
            - 'pnorm': minimize Lp-th matrix norm.
            Equivalent to abs when p = 1.
            - 'norm': minimize different type of matrix norm
            (see norm_type below).
    sample_tol : float
        Minimum eigenvalue allowed for feature-knockoff covariance 
        matrix. Keep this small but nonzero (1e-5) to prevent numerical errors.
    verbose : bool
        If True, prints progress over time
    rec_prop : float
        The proportion of knockoffs to recycle (see Barber and Candes 2018,
        https://arxiv.org/abs/1602.03574). If method = 'mvr', then S_generation 
        takes this into account and should increase the power of recycled knockoffs.    sparsely-correlated, high-dimensional settings.
    kwargs : dict
        Other kwargs for S-matrix solvers.
    """

    def __init__(
        self,
        X,
        mu=None,
        Sigma=None,
        invSigma=None,
        groups=None,
        sample_tol=1e-5,
        S=None,
        method=None,
        verbose=False,
        **kwargs,
    ):

        # Save parameters with defaults
        self.X = X
        self.n = X.shape[0]
        self.p = X.shape[1]
        if mu is None:
            mu = X.mean(axis=0)
        self.mu = mu
        if Sigma is None:
            Sigma, invSigma = utilities.estimate_covariance(X, tol=1e-2)
        self.Sigma = Sigma
        if invSigma is None:
            invSigma = utilities.chol2inv(Sigma)
        self.invSigma = invSigma
        if groups is None:
            groups = np.arange(1, self.p + 1, 1)
        self.groups = groups
        self.sample_tol = sample_tol
        self.verbose = verbose

        # Save S information and possibly compute S matrix
        self.S = S
        self.method = method
        self.kwargs = kwargs
        if self.S is None:
            if self.verbose:
                print(f"Computing knockoff S matrix...")
            self.S = smatrix.compute_smatrix(
                Sigma=self.Sigma, groups=self.groups, method=self.method, **self.kwargs
            )

    def fetch_S(self):
        return self.S

    def sample_knockoffs(self):
        """ Samples knockoffs. returns n x p knockoff matrix."""
        self.check_PSD_condition(self.Sigma, self.S)
        self.Xk = produce_MX_gaussian_knockoffs(
            X=self.X,
            mu=self.mu,
            invSigma=self.invSigma,
            S=self.S,
            sample_tol=self.sample_tol,
            copies=1,
            verbose=self.verbose,
        )[:, :, 0]
        return self.Xk


def produce_FX_knockoffs(X, invSigma, S, copies=1):
    """
    See equation (1.4) of https://arxiv.org/pdf/1404.5609.pdf
    """

    # Calculate C matrix
    n, p = X.shape
    # invSigma_S = np.dot(invSigma, S)
    CTC = 2 * S - np.dot(S, np.dot(invSigma, S))
    C = scipy.linalg.cholesky(CTC)

    # Calculate U matrix
    Q, _ = scipy.linalg.qr(np.concatenate([X, np.zeros((n, p))], axis=1))
    U = Q[:, p : 2 * p]

    # Randomize if copies > 1
    knockoff_base = np.dot(X, np.eye(p) - np.dot(invSigma, S))
    if copies > 1:
        knockoffs = []
        for j in range(copies):

            # Multiply U by random orthonormal matrix
            Qj, _ = scipy.linalg.qr(np.random.randn(p, p))
            Uj = np.dot(U, Qj)

            # Calculate knockoffs
            knockoff_j = knockoff_base + np.dot(Uj, C)
            knockoffs.append(knockoff_j)
    else:
        # Calculate knockoffs and return
        knockoffs = [(knockoff_base + np.dot(U, C))]

    knockoffs = np.stack(knockoffs, axis=-1)
    return knockoffs


class FXSampler(KnockoffSampler):
    """ 
    Samples FX knockoffs. See the GaussianSampler documentation 
    for description of the arguments.
    """

    def __init__(
        self,
        X,
        groups=None,
        sample_tol=1e-5,
        S=None,
        method=None,
        verbose=False,
        **kwargs,
    ):

        # Save data
        self.X = X.copy()
        self.n = X.shape[0]
        self.p = X.shape[1]
        if self.n < 2 * self.p:
            raise np.linalg.LinAlgError(
                f"FX knockoffs can't be generated with n ({self.n}) < 2p ({2*self.p})"
            )
        self.Sigma = np.dot(self.X.T, self.X)
        self.invSigma = utilities.chol2inv(self.Sigma)
        kwargs.pop("Sigma", None)
        kwargs.pop("invSigma", None)

        # Other parameters
        if groups is None:
            groups = np.arange(1, self.p + 1, 1)
        self.groups = groups
        self.sample_tol = sample_tol
        self.verbose = verbose

        # Save S information and possibly compute S matrix
        self.S = S
        self.method = method
        self.kwargs = kwargs
        if self.S is None:
            if self.verbose:
                print(f"Computing knockoff S matrix...")
            self.S = smatrix.compute_smatrix(
                Sigma=self.Sigma, groups=self.groups, method=self.method, **self.kwargs
            )

    def fetch_S(self):
        """ Rescales S to the same scale as the initial X input """
        return self.S

    def sample_knockoffs(self):
        """ Samples knockoffs. returns n x p knockoff matrix."""
        self.check_PSD_condition(self.Sigma, self.S)
        self.Xk = produce_FX_knockoffs(
            X=self.X, invSigma=self.invSigma, S=self.S, copies=1,
        )[:, :, 0]
        return self.Xk
