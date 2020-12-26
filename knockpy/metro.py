""" 
    The metropolized knockoff sampler for an arbitrary probability density
    and graphical structure using covariance-guided proposals.

    See https://arxiv.org/abs/1903.00434 for a description of the algorithm
    and proof of validity and runtime.

    This code was based on initial code written by Stephen Bates in October
    2019, which was released in combination with https://arxiv.org/abs/1903.00434.
"""

# The basics
import sys
import copy
import numpy as np
import scipy as sp
import scipy.special
import itertools
from functools import reduce
from scipy import stats
from . import utilities, knockoffs, dgp

# Network and UGM tools
import networkx as nx
from networkx.algorithms.approximation import treewidth
from .knockoffs import KnockoffSampler
from . import knockoffs, smatrix, constants

# Logging
import warnings
from tqdm import tqdm


def gaussian_log_likelihood(X, mu, var):
    """
    Somehow this is faster than scipy
    """
    result = -1 * np.power(X - mu, 2) / (2 * var)
    result += np.log(1 / np.sqrt(2 * np.pi * var))
    return result


def t_log_likelihood(X, df_t):
    """
    UNNORMALIZED t loglikelihood.
    This is also faster than scipy
    """
    result = np.log(1 + np.power(X, 2) / df_t)
    result = -1 * result * (df_t + 1) / 2
    return result


def get_ordering(T):
    """ 
    Takes a junction tree and returns a variable ordering for the metro
    knockoff sampler. The code from this function is adapted from
    the code distributed with https://arxiv.org/abs/1903.00434.

    Parameters
    ----------
    T : A networkx graph that is a junction tree.
    Nodes must be sets with elements 0,...,p-1. 

    Returns
    -------
    order : a numpy array with unique elements 0,...,p-1
    active_frontier : list of lists
        a list of length p gwhere entry j is the set of entries > j 
        that are in V_j. This specifies the conditional independence structure
        of a joint covariate distribution. See page 34 of
        https://arxiv.org/abs/1903.00434.
    """
    # Initialize
    T = T.copy()
    order = []
    active_frontier = []

    while T.number_of_nodes() > 0:
        # Loop through leaf nodes
        gen = (x for x in T.nodes() if T.degree(x) <= 1)
        active_node = next(gen)

        # Parent nodes of leaf nodes
        # active_vars get added to the order in this step
        # activated set are just the variables in the active node
        parents = list(T[active_node].keys())
        if len(parents) == 0:
            active_vars = set(active_node)
            activated_set = active_vars.copy()
        else:
            active_vars = set(active_node.difference(parents[0]))
            activated_set = active_vars.union(parents[0]).difference(set(order))
        for i in list(active_vars)[::-1]:  # This line was changed too
            order += [i]
            frontier = list(activated_set.difference(set(order)))
            active_frontier += [frontier]
        T.remove_node(active_node)

    return [np.array(order), active_frontier]


class MetropolizedKnockoffSampler(KnockoffSampler):
    """
    A metropolized knockoff sampler for arbitrary random variables
    using covariance-guided proposals.

    Group knockoffs are not yet supported.

    Parameters
    ----------
    lf : function
        log-probability density. This function should take a ``(n, p)``-shaped
        numpy array (n independent samples of a p-dimensional vector) and return
        a ``(n,)`` shaped array of log-probabilities. This can also be supplied 
        as ``None`` if cliques and log-potentials are supplied.
    X : np.ndarray
        the ``(n, p)``-shaped design matrix
    Xk : np.ndarray
        the ``(n, p)``-shaped matrix of knockoffs
    mu : np.ndarray
        The (estimated) mean of X. Exact FDR control is maintained
        even when this vector is incorrect. Defaults to the mean of X,
        e.g., ``X.mean(axis=0)``.
    Sigma : np.ndarray
        ``(p, p)``-shaped covariance matrix of the features. If ``None``, this
        is estimated using the data using a naive method to ensure compatability
        with the proposals. Exact FDR control is maintained even when Sigma 
        is incorrect.
    undir_graph : np.ndarray or nx.Graph
        An undirected graph specifying the conditional independence
        structure of the data-generating process. This must be specified 
        if either of the ``order`` or ``active_frontier`` params
        are not specified. One of two options:
        - A networkx undirected graph object
        - A ``(p, p)``-shaped numpy array, where nonzero elements represent edges.
    order : np.ndarray
        A ``p``-length numpy array specifying the ordering to sample the variables.
        Should be a vector with unique entries 0,...,p-1.
    active_fontier : A list of lists of length p where entry j is the set of
        entries > j that are in V_j. This specifies the conditional independence 
        structure of the distribution given by lf. See page 34 of the paper.
    gamma : float
        A tuning parameter to increase / decrease the acceptance ratio.
        See appendix F.2. Defaults to 0.999.
    buckets : np.ndarray or list
        A list or array of discrete values that X can take.
        Covariance-guided proposals will be rounded to these values. 
        If ``None``, Metro assumes the domain of each feature is all
        real numbers.
    kwargs : dict
        kwargs to pass to the ``smatrix.compute_smatrix`` method for 
        sampling proposals.

    Attributes
    ----------
    order : np.ndarray
        ``(p,)``-shaped array of indices which reorders ``X`` into the
        order for sampling knockoffs. 
    inv_order : np.ndarray
        ``(p,)``-shaped array of indices which takes a set of variables
        which have been reordered for metropolized sampling and returns
        them to their initial order. For example,
        ``X == X[:, self.order][:, self.inv_order]``.
    X : np.ndarray
        ``(n, p)`` design matrix reordered according to the order for
        sampling knockoffs
    X_prop : np.ndarray
        ``(n, p)``-shaped array of knockoff proposals
    Xk : np.ndarray
        the ``(n, p)``-shaped array of knockoffs
    acceptances : np.ndarray
        a ``(n, p)``-shaped boolean array where ``acceptances[i, j] == 1``
        indicates that ``X_prop[i, j]`` was accepted.
    final_acc_probs : np.ndarray
        a ``(n, p)``-shaped array where ``final_acc_probs[i, j]`` is the 
        acceptance probability for ``X_prop[i, j]``.
    Sigma : np.ndarray
        the ``(p, p)``-shaped estimated covariance matrix of ``X``. The 
        class constructor guarantees this is compatible with the conditional
        independence structure of the data.
    S : np.ndarray
        the ``(p, p)``-shaped knockoff S-matrix used to generate the
        covariance-guided proposals.

    Notes
    -----
    All attributes of the MetropolizedKnockoffSampler are stored in the 
    order that knockoffs are sampled, NOT the order that variables are 
    initially passed in. For example, the ``X`` attribute will not necessarily
    equal the ``X`` argument: instead, ``self.X = X[:, self.order]``. To reorder
    attributes to the initial order of the ``X`` argument,
    use the syntax ``self.attribute[:, self.inv_order]``.
    """
    def __init__(
        self,
        lf,
        X,
        mu=None,
        Sigma=None,
        undir_graph=None,
        order=None,
        active_frontier=None,
        gamma=0.999,
        metro_verbose=False,
        cliques=None,
        log_potentials=None,
        buckets=None,
        **kwargs,
    ):

        # Random params
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.gamma = gamma
        self.metro_verbose = metro_verbose  # Controls verbosity
        self.F_queries = 0  # Counts how many queries we make
        self.buckets = buckets

        # Possibly estimate mean, cov matrix
        if mu is None:
            mu = X.mean(axis=0)
        V = Sigma  # Improves readability slightly
        cov_est = False
        if V is None:
            cov_est = True
            V, Q = utilities.estimate_covariance(X, tol=1e-3, shrinkage=None)

        # Possibly learn order / active frontier
        if order is None or active_frontier is None:
            if undir_graph is None:
                raise ValueError(
                    f"If order OR active_frontier are not provided, you must specify the undir_graph"
                )
            # Convert to nx
            if isinstance(undir_graph, np.ndarray):
                undir_graph = nx.Graph(undir_graph != 0)
            # Run junction tree algorithm
            self.width, self.T = treewidth.treewidth_decomp(undir_graph)
            order, active_frontier = get_ordering(self.T)

        # Undirected graph must be existent in this case
        if "invSigma" in kwargs:
            Q = kwargs.pop("invSigma")
        else:
            # This is more numerically stable for super sparse Q
            Q = np.linalg.inv(V)
        if undir_graph is not None:
            warnings.filterwarnings("ignore")
            mask = nx.to_numpy_matrix(undir_graph)
            warnings.resetwarnings()
            np.fill_diagonal(mask, 1)
            # Handle case where the graph is entirely dense
            if (mask == 0).sum() > 0 and not cov_est:
                max_nonedge = np.max(np.abs(Q[mask == 0]))
                if max_nonedge > 1e-2:
                    warnings.warn(
                        f"Precision matrix Q is not compatible with undirected graph (nonedge has value {max_nonedge}); naively forcing values to zero"
                    )
            elif cov_est:
                Q[mask == 0] = 0
                V = utilities.chol2inv(Q)

        # Save order and inverse order
        self.order = order
        self.inv_order = order.copy()
        for i, j in enumerate(order):
            self.inv_order[j] = i

        # Re-order the variables: the log-likelihood
        # function (lf) is reordered in a separate method
        self.X = X[:, self.order].astype(np.float32)
        self.unordered_lf = lf

        # Re-order the cliques
        # (internal order, not external order)
        self.log_potentials = log_potentials
        if cliques is not None:
            self.cliques = []
            for clique in cliques:
                self.cliques.append(self.inv_order[clique])
        else:
            self.cliques = None

        # Create clique dictionaries. This maps variable i
        # to a list of two-length tuples.
        #   - The first element is the clique_key, which can
        #     be used to index into log_potentials.
        #   - The second element is the actual clique.
        if self.cliques is not None and self.log_potentials is not None:
            self.clique_dict = {j: [] for j in range(self.p)}
            for clique_key, clique in enumerate(self.cliques):
                for j in clique:
                    self.clique_dict[j].append((clique_key, clique))
        else:
            self.clique_dict = None

        # Re-order active frontier
        self.active_frontier = []
        for i in range(len(order)):
            self.active_frontier += [[self.inv_order[j] for j in active_frontier[i]]]

        # Re-order mu
        self.mu = mu[self.order].reshape(1, -1)

        # If mu == 0, then we can save lots of time
        if np.all(self.mu == 0):
            self._zero_mu_flag = True
        else:
            self._zero_mu_flag = False

        # Re-order sigma
        self.V = V[self.order][:, self.order]
        self.Sigma = V
        self.Q = Q[self.order][:, self.order]

        # Possibly reorder S if it's in kwargs
        if "S" in kwargs:
            if kwargs["S"] is not None:
                S = kwargs["S"]
                kwargs["S"] = S[self.order][:, self.order]

        # Create proposal parameters
        self.create_proposal_params(**kwargs)

    def lf_ratio(self, X, Xjstar, j):
        """
        Calculates the log of the likelihood ratio
        between two observations: X where X[:,j] 
        is replaced with Xjstar, divided by the likelihood
        of X. This is equivalent to (but often faster) than:

        >>> ld_obs = self.lf(X)
        >>> Xnew = X.copy()
        >>> Xnew[:, j] = Xjstar
        >>> ld_prop = self.lf(Xnew)
        >>> ld_ratio = ld_prop - ld_obs

        When node potentials have been passed, this is much faster
        than calculating the log-likelihood function and subtracting.

        :param X: a n x p matrix of observations
        :param Xjstar: New observations for column j of X
        :param j: an int between 0 and p - 1, telling us which column to replace
        """

        # Just return the difference in lf if we don't have
        # access to cliques
        if self.clique_dict is None or self.log_potentials is None:
            # Log-likelihood 1
            ld_obs = self.lf(X)
            # New likelihood with Xjstar
            Xnew = X.copy()
            Xnew[:, j] = Xjstar
            ld_prop = self.lf(Xnew)
            ld_ratio = ld_prop - ld_obs

        # If we have access to cliques, we can just compute log-potentials
        else:
            cliques = self.clique_dict[j]
            ld_ratio = np.zeros(self.n)

            # Loop through cliques
            for clique_key, clique in cliques:
                # print(f"At clique_key {clique_key},clique {clique}, j={j}")
                # print(f"Orig clique is {self.order[clique]}")
                orig_clique = self.order[clique]  # Original ordering

                # Clique representation(s) of X
                Xc = X[:, clique]
                Xcstar = Xc.copy()

                # Which index corresponds to index j in the clique
                new_j = np.where(orig_clique == self.order[j])[0]
                Xcstar[:, new_j] = Xjstar.reshape(-1, 1)

                # Calculate log_potential difference
                ld_ratio += self.log_potentials[clique_key](Xcstar).reshape(-1)
                ld_ratio -= self.log_potentials[clique_key](Xc).reshape(-1)

        return ld_ratio

    def lf(self, X):
        """ Reordered likelihood function """
        return self.unordered_lf(X[:, self.inv_order])

    def center(self, M, active_inds=None):
        """
        Centers an n x j matrix M. For mu = 0, does not perform
        this computation, which actually is a bottleneck
        for large n and p.
        """
        if self._zero_mu_flag:
            return M
        elif active_inds is None:
            return M - self.mu[:, 0 : M.shape[1]]
        else:
            return M - self.mu[:, active_inds]

    def create_proposal_params(self, **kwargs):
        """
        Constructs the covariance-guided proposal. 

        Parameters
        ----------
        kwargs : dict
            kwargs for the ``smatrix.compute_smatrix`` function
        """

        # Find the optimal S matrix. In general, we should set a
        # fairly high tolerance to avoid numerical errors.
        kwargs["tol"] = kwargs.pop(
            "tol",
            min(
                constants.METRO_TOL,
                np.linalg.eigh(self.V)[0].min()/10
            )
        )
        self.S = smatrix.compute_smatrix(Sigma=self.V, **kwargs)
        self.G = np.concatenate(
            [
                np.concatenate([self.V, self.V - self.S]),
                np.concatenate([self.V - self.S, self.V]),
            ],
            axis=1,
        )
        self.invG = None  # We don't compute this unless we have to later

        # Check for PSD-ness
        minSeig = np.min(np.diag(self.S))
        min2VSeig = np.linalg.eigh(2 * self.V - self.S)[0].min()
        if minSeig < 0:
            raise np.linalg.LinAlgError(f"Minimum eigenvalue of S is {minSeig}")
        if min2VSeig < 0:
            raise np.linalg.LinAlgError(
                f"Minimum eigenvalue of 2 Sigma - S is {min2VSeig}"
            )
        if self.metro_verbose:
            print(f"Minimum eigenvalue of S is {minSeig}")
            print(f"Minimum eigenvalue 2V-S is {min2VSeig}")

        # Efficiently calculate p inverses of subsets
        # of feature-knockoff covariance matrix.
        # This uses Cholesky decompositions for numerical
        # stability
        # Cholesky decomposition of Sigma
        self.invSigma = self.Q.copy()
        self.L = np.linalg.cholesky(self.V)
        initial_error = np.max(np.abs(self.V - np.dot(self.L, self.L.T)))

        # Suppose X sim N(mu, Sigma) and we have proposals X_{1:j-1}star
        # Then the conditional mean of the proposal Xjstar
        # is muj + mean_transform @ [X - mu, X_{1:j-1}^* - mu_{1:j-1}]
        # where the brackets [] denote vector concatenation.
        # This code calculates those mean transforms and the
        # conditional variances.
        self.cond_vars = np.zeros(self.p)
        self.mean_transforms = []

        # Possibly log
        if self.metro_verbose:
            print(f"Metro starting to compute proposal parameters...")
            j_iter = tqdm(range(0, self.p))
        else:
            j_iter = range(0, self.p)

        # Loop through and compute
        # j corresponds to the jth knockoff variable
        for j in j_iter:

            # G up to and excluding knockoff j
            Gprej = self.G[0 : self.p + j, 0 : self.p + j]
            gammaprej = Gprej[-1, 0:-1]  # marginal corrs btwn knockoff j + others
            sigma2prej = Gprej[-1, -1]

            # 1. Compute inverse Sigma
            if j > 0:

                # At this point, we want L to be the cholesky
                # factor of Gprej, but it is the cholesky factor
                # of Gprej{j-1}. Therefore we perform a rank-1
                # update.
                # The c below is computed previously:
                # c = sp.linalg.solve_triangular(
                #   a=self.L, b=gammaprej, lower=True
                # )
                d = np.sqrt(sigma2prej - np.dot(c, c))

                # Concatenate new row [c,d] to L
                new_row = np.concatenate([c, np.array([d])]).reshape(1, -1)
                self.L = np.concatenate([self.L, np.zeros((self.p + j - 1, 1))], axis=1)
                self.L = np.concatenate([self.L, new_row], axis=0)

            # Check for numerical instabilities
            # diff = Gprej - np.dot(self.L, self.L.T)
            # max_error = np.max(np.abs(diff))
            # if max_error > 10 * initial_error:
            #     # Correct
            #     print(
            #         f"Maximum error is {max_error} > 10x init error, recomputing L for p={self.p}, j={j}"
            #     )
            #     self.L = np.linalg.cholesky(Gprej)

            # 2. Compute conditional variance
            # This subset of G includes knockoff j
            Ginclj = self.G[0 : self.p + j + 1, 0 : self.p + j + 1]
            gammainclj = Ginclj[-1, 0:-1]  # marginal corrs btwn knockoff j + others
            marg_var = Ginclj[-1, -1]

            # Cholesky trick: Sigma[j,j] - ||c||_2^2
            # is the conditional variance
            c = sp.linalg.solve_triangular(
                a=self.L, b=gammainclj, lower=True, overwrite_b=False
            )
            self.cond_vars[j] = marg_var - np.dot(c, c)

            # Sanity check
            msg = "This is likely a numerical error --- try increasing the tol kwarg."
            if self.cond_vars[j] < 0:
                raise ValueError(f"Cond_vars[{j}]={self.cond_vars[j]} < 0. {msg}")
            elif self.cond_vars[j] > marg_var:
                raise ValueError(
                    f"Cond_vars[{j}]={self.cond_vars[j]} > marginal variance {marg_var}. {msg}"
                )

            # Mean transform
            mean_transform = sp.linalg.solve_triangular(
                a=self.L.T, b=c, lower=False, overwrite_b=False
            ).reshape(1, -1)
            self.mean_transforms.append(mean_transform)

    def fetch_S(self):
        return self.S[self.inv_order][:, self.inv_order]

    def fetch_proposal_params(self, X, prev_proposals):
        """
        Returns mean and variance of proposal j given X and
        previous proposals. Both ``X`` and ``prev_proposals``
        must be in the order used to sample knockoff variables.
        """

        # Infer j from prev_proposals
        if prev_proposals is not None:
            j = prev_proposals.shape[-1]
        else:
            j = 0

        # First p coordinates of cond_mean
        # X is n x p
        # self.mu is 1 x p
        # self.mean_transforms[j] is 1 x p + j
        # However, this cond mean only depends on
        # the active variables + [0:j], so to save
        # computation, we only compute that dot
        active_inds = list(range(j + 1)) + self.active_frontier[j]
        cond_mean = np.dot(
            self.center(X[:, active_inds], active_inds),
            self.mean_transforms[j][:, active_inds].T,
        )

        # Second p coordinates of cond_mean
        if j != 0:
            # prev_proposals is n x j
            # self.mean_transforms[j] is 1 x p + j
            # self.mu is 1 x p
            cond_mean2 = np.dot(
                self.center(prev_proposals), self.mean_transforms[j][:, self.p :].T
            )
            # Add together
            cond_mean += cond_mean2

        # Shift and return
        cond_mean += self.mu[0, j]
        return cond_mean, self.cond_vars[j]

    def fetch_cached_proposal_params(self, Xtemp, x_flags, j):
        """
        Same as above, but uses caching to speed up computation.
        This caching can be cheap (if self.cache is False) or
        extremely expensive (if self.cache is True) in terms of
        memory.
        """

        # Conditional mean only depends on these inds
        active_inds = list(range(j + 1)) + self.active_frontier[j]

        # Calculate conditional means from precomputed products
        if self.cache:
            cond_mean = np.where(
                x_flags[:, active_inds],
                self.cached_mean_obs_eq_prop[j],
                self.cached_mean_obs_eq_obs[j],
            ).sum(axis=1)
        else:
            active_inds = list(range(j + 1)) + self.active_frontier[j]
            cond_mean = np.dot(
                self.center(Xtemp[:, active_inds], active_inds),
                self.mean_transforms[j][:, active_inds].T,
            ).reshape(-1)

        # Add the effect of conditioning on the proposals
        cond_mean += self.cached_mean_proposals[j]
        return cond_mean, self.cond_vars[j]

    def q_ll(self, Xjstar, X, prev_proposals, cond_mean=None, cond_var=None):
        """
        Calculates the log-likelihood of a proposal Xjstar given X 
        and the previous proposals.
        Xjstar : np.ndarray
            ``(n,)``-shaped numpy array of values to evaluate the proposal
            likelihood at.
        X : np.ndarray
            ``(n, p)``-shaped array of observed data, in the order used to
            sample knockoff variables.
        prev_proposals : np.ndarray
            ``(n, j-1)``-shaped array of previous proposals, in the order
            used to sample knockoff variables. If None, assumes j = 0.
        """
        if cond_mean is None or cond_var is None:
            cond_mean, cond_var = self.fetch_proposal_params(
                X=X, prev_proposals=prev_proposals
            )
        # Continuous case
        if self.buckets is None:
            return gaussian_log_likelihood(Xjstar, cond_mean.reshape(-1), cond_var)
        else:
            bucket_probs = gaussian_log_likelihood(
                X=self.buckets.reshape(1, -1),
                mu=cond_mean.reshape(-1, 1),
                var=cond_var,
            )
            bucket_log_probs = scipy.special.log_softmax(
                bucket_probs.astype(np.float32), axis=-1
            )
            flags = Xjstar.reshape(-1, 1) == self.buckets.reshape(1, -1)
            out = bucket_log_probs[flags]
            return out

    def sample_proposals(
        self, X, prev_proposals, cond_mean=None, cond_var=None,
    ):
        """
        Samples a continuous or discrete proposal given the design
        matrix and the previous proposals. Can pass in the conditional
        mean and variance of the new proposals, if cached, to save
        computation.
        """
        # These will be compatible as long as Sigma is
        if cond_mean is None or cond_var is None:
            cond_mean, cond_var = self.fetch_proposal_params(
                X=X, prev_proposals=prev_proposals
            )
        # Continuous sampling
        if self.buckets is None:
            proposal = np.sqrt(cond_var) * np.random.randn(self.n, 1) + cond_mean

        # Discrete sampling
        if self.buckets is not None:
            # Cumulative probability buckets
            bucket_probs = gaussian_log_likelihood(
                X=self.buckets, mu=cond_mean, var=cond_var,
            )
            bucket_probs = scipy.special.softmax(
                bucket_probs.astype(np.float32), axis=-1
            )
            cuml_bucket_probs = bucket_probs.cumsum(axis=-1)

            # Sample independently n times from buckets
            unifs = np.random.uniform(size=(self.n, 1))
            inds = np.argmin(cuml_bucket_probs < unifs, axis=-1)
            proposal = self.buckets[inds]

        return proposal.reshape(-1)

    def _get_key(self, x_flags, j):
        """
        Fetches key for dp dicts
        """
        inds = list(
            self.active_frontier[j]
        )  # list(set(self.active_frontier[j]).union(set([j])))
        arr_key = x_flags[0, inds]
        key = arr_key.tobytes()
        return key

    def _key2bool(self, key):
        """
        Takes a key from dp dicts
        and turns it back into a boolean array.
        """
        return np.frombuffer(key, dtype=bool)

    def _create_Xtemp(self, x_flags, j):
        """
        Returns a n x p array Xtemp which effectively does:
        Xtemp = self.X.copy()
        Xtemp[x_flags == 1] = self.X_prop[x_flags == 1].copy()

        TODO: make this so it copies less. This may require 
        C code.
        """

        # Informative error
        if x_flags[:, 0:j].sum() > 0:
            raise ValueError(
                f"x flags are {x_flags} for j={j}, strange because they should be zero before j"
            )

        Xtemp = np.where(x_flags, self.X_prop, self.X)
        return Xtemp

    def log_q12(self, x_flags, j):
        """
        Computes q1 and q2 as specified by page 33 of the paper.
        """

        # Temporary vector of Xs for query
        Xtemp = self._create_Xtemp(x_flags, j)

        # Precompute cond_means for log_q2
        cond_mean2, cond_var = self.fetch_cached_proposal_params(
            Xtemp=Xtemp, x_flags=x_flags, j=j,
        )
        # Adjust cond_mean for q1
        diff = self.X_prop[:, j] - Xtemp[:, j]
        adjustment = self.mean_transforms[j][:, j] * (diff)
        cond_mean1 = cond_mean2.reshape(-1) + adjustment

        ### Continuous case
        if self.buckets is None:
            # q2 is:
            # Pr(Xjstar = xjstar | X = Xtemp, tildeX_{1:j-1}, Xstar_{1:j-1})
            log_q2 = gaussian_log_likelihood(
                X=self.X_prop[:, j], mu=cond_mean2.reshape(-1), var=cond_var,
            )

            # q1 is:
            # Pr(Xjstar = Xtemp[j] | Xj = xjstar, X_{-j} = X_temp_{-j}, tildeX_{1:j-1}, Xstar_{1:j-1})
            log_q1 = gaussian_log_likelihood(
                X=Xtemp[:, j], mu=cond_mean1.reshape(-1), var=cond_var,
            )
        else:
            # Terms are same as before
            log_q2 = self.q_ll(
                Xjstar=self.X_prop[:, j],
                X=None,
                prev_proposals=None,
                cond_mean=cond_mean2,
                cond_var=cond_var,
            )
            log_q1 = self.q_ll(
                Xjstar=Xtemp[:, j],
                X=None,
                prev_proposals=None,
                cond_mean=cond_mean1,
                cond_var=cond_var,
            )

        return log_q1, log_q2, Xtemp

    def compute_F(self, x_flags, j):
        """
        Computes the F function from Page 33 pf the paper: 
        Pr(tildeXj=tildexj, Xjstar=xjstar | Xtemp, tildeX_{1:j-1}, Xjstar_{1:j-1})
        Note that tildexj and xjstar are NOT inputs because they do NOT change
        during the junction tree DP process.
        """

        # Get key, possibly return cached result
        key = self._get_key(x_flags, j)
        if key in self.F_dicts[j]:
            return self.F_dicts[j][key]
        else:
            self.F_queries += 1

        # q1/q2 terms
        log_q1, log_q2, Xtemp = self.log_q12(x_flags, j)

        # Acceptance mask and probabilities: note that
        # the flag for accepting / rejecting comes from the
        # TRUE knockoffs (e.g. self.acceptances)
        if key in self.acc_dicts[j]:
            acc_probs = self.acc_dicts[j][key]
        else:
            # Pass extra parameters to avoid repeating computation
            acc_probs = self.compute_acc_prob(
                x_flags=x_flags, j=j, log_q1=log_q1, log_q2=log_q2, Xtemp=Xtemp
            )
        mask = self.acceptances[:, j] == 1
        result = log_q2 + mask * np.log(acc_probs) + (1 - mask) * np.log(1 - acc_probs)

        # Cache
        self.F_dicts[j][key] = result

        # Return
        return result

    def compute_acc_prob(
        self, x_flags, j, log_q1=None, log_q2=None, Xtemp=None,
    ):
        """
        Computes acceptance probability for variable ``j``
        given a particular rejection pattern ``x_flags``.

        Mathematically, this is:
        Pr(tildeXj = Xjstar | Xtemp, Xtilde_{1:j-1}, Xstar_{1:j})
        """

        # Get key, possibly return cached result
        key = self._get_key(x_flags, j)
        if key in self.acc_dicts[j]:
            return self.acc_dicts[j][key]

        # 1. q1, q2 ratio
        if log_q1 is None or log_q2 is None:
            log_q1, log_q2, Xtemp = self.log_q12(x_flags, j)
        lq_ratio = log_q1 - log_q2

        # Possibly ceate X temp variable
        if Xtemp is None:
            Xtemp = self._create_Xtemp(x_flags, j)

        # 2. Density ratio
        ld_ratio = self.lf_ratio(X=Xtemp, Xjstar=self.X_prop[:, j], j=j,)
        # # a. According to pattern
        # ld_obs = self.lf(Xtemp)
        # # b. When Xj is not observed
        # Xtemp_prop = Xtemp.copy()
        # Xtemp_prop[:, j] = self.X_prop[:, j]
        # ld_prop = self.lf(Xtemp_prop)
        # ld_ratio = ld_prop - ld_obs

        # Delete to save memory
        del Xtemp

        # 3. Calc ln(Fk ratios) for k < j. These should be 0 except
        # when k < j and j in Vk, which is why we loop through
        # affected variables.
        # Numerator for these ratios use different flags
        Fj_ratio = np.zeros(self.n)
        x_flags_num = x_flags.copy()
        x_flags_num[:, j] = 1

        # Loop through
        for j2 in self.affected_vars[j]:

            # Numerator
            num_key = self._get_key(x_flags_num, j2)
            if num_key in self.F_dicts[j2]:
                Fj2_num = self.F_dicts[j2][num_key]
            else:
                Fj2_num = self.compute_F(x_flags_num, j2)

            # Denominator uses same flags
            denom_key = self._get_key(x_flags, j2)
            if denom_key in self.F_dicts[j2]:
                Fj2_denom = self.F_dicts[j2][denom_key]
            else:
                Fj2_denom = self.compute_F(x_flags, j2)

            # Add ratio to Fj_ratio
            Fj_ratio = Fj_ratio + Fj2_num - Fj2_denom

        # Put it all together and multiply by gamma
        # Fast_exp function is helpful for profiling
        # (to see how much time is spent here)
        def fast_exp(ld_ratio, lq_ratio, Fj_ratio):
            return np.exp((ld_ratio + lq_ratio + Fj_ratio).astype(np.float32))

        acc_prob = self.gamma * np.minimum(1, fast_exp(ld_ratio, lq_ratio, Fj_ratio))

        # Clip to deal with floating point errors
        acc_prob = np.minimum(self.gamma, np.maximum(self.clip, acc_prob))

        # Make sure the degenerate case has been computed
        # correctly
        if x_flags[:, j].sum() > 0:
            if acc_prob[x_flags[:, j] == 1].mean() <= self.gamma:
                msg = f"At step={self.step}, j={j}, we have"
                msg += f"acc_prob = {acc_prob} but x_flags[:, j]={x_flags[:, j]}"
                msg += f"These accetance probs should be ~1"
                raise ValueError(msg)

        # Cache and return
        self.acc_dicts[j][key] = acc_prob
        return acc_prob

    def cache_conditional_proposal_params(self, verbose=False, expensive_cache=True):
        """
        Caches some of the conditional means for Xjstar | Xtemp.
        If expensive_cache = True, this will be quite memory intensive
        in order to achieve a 2-3x speedup. Otherwise, achieves a
        a 20-30% speedup at a more modest memory cost.
        """

        # Cache conditional means
        if verbose:
            print(f"Metro beginning to cache conditional means...")
            j_iter = tqdm(range(self.p))
        else:
            j_iter = range(self.p)

        # Precompute centerings
        centX = self.center(self.X)
        centX_prop = self.center(self.X_prop)
        self.cached_mean_obs_eq_obs = [None for _ in range(self.p)]
        self.cached_mean_obs_eq_prop = [None for _ in range(self.p)]
        self.cached_mean_proposals = [None for _ in range(self.p)]
        for j in j_iter:

            # We only need to store the coordinates along the active
            # inds which saves some memory
            active_inds = list(range(j + 1)) + self.active_frontier[j]

            # Cache some precomputed conditional means
            # a. Cache the effect of conditioning on Xstar = self.X_prop
            # This is very cheap
            self.cached_mean_proposals[j] = np.dot(
                self.center(self.X_prop[:, 0:j]), self.mean_transforms[j][:, self.p :].T
            ).reshape(-1)

            # b/c: Possibly cache the effect of conditioning on X = self.X / self.X_prop
            # This is very memory intensive
            if expensive_cache:
                # a. Cache the effect of conditiong on X = self.X
                cache_obs = (
                    centX[:, active_inds]
                    * self.mean_transforms[j][:, 0 : self.p][:, active_inds]
                ).astype(np.float32)
                self.cached_mean_obs_eq_obs[j] = cache_obs
                # b. Cache the effect of conditioning on X = self.X_prop
                cache_prop = (
                    centX_prop[:, active_inds]
                    * self.mean_transforms[j][:, 0 : self.p][:, active_inds]
                ).astype(np.float32)
                self.cached_mean_obs_eq_prop[j] = cache_prop

    def sample_knockoffs(self, clip=1e-5, cache=None):
        """
        Samples knockoffs using the metropolized knockoff
        sampler.

        Parameters
        ----------
        clip : float
            To provide numerical stability, we make the minimum
            acceptance probability clip. If ``clip=0``, some 
            acceptance probabilities may become negative due to
            floating point errors.
        cache : bool
            If True, uses a very memory intensive caching system
            to get a 2-3x speedup when calculating conditional means
            for the proposals. Defaults to true if n * (p**2) < 1e9.

        Returns
        -------
        Xk : np.ndarray
            A ``(n, p)``-shaped knockoff matrix in the original order 
            the variables were passed in.
        """

        # Save clip constant for later
        self.clip = clip
        num_params = self.n * (self.p ** 2)
        if cache is not None:
            self.cache = cache
        else:
            self.cache = num_params < 1e9

        # Possibly log
        if self.metro_verbose:
            if self.cache:
                print(
                    f"Metro will use memory expensive caching for 2-3x speedup, storing {num_params} params"
                )
            else:
                print(f"Metro will not cache cond_means to save a lot of memory")

        # Dynamic programming approach: store acceptance probs
        # as well as Fj values (see page 33)
        self.acc_dicts = [{} for j in range(self.p)]
        self.F_dicts = [{} for j in range(self.p)]

        # Locate previous terms affected by variable j
        self.affected_vars = [[] for k in range(self.p)]
        for j, j2 in itertools.product(range(self.p), range(self.p)):
            if j in self.active_frontier[j2]:
                self.affected_vars[j] += [j2]

        # Store pattern of TRUE acceptances / rejections
        self.acceptances = np.zeros((self.n, self.p)).astype(np.bool)
        self.final_acc_probs = np.zeros((self.n, self.p))

        # Proposals
        self.X_prop = np.zeros((self.n, self.p)).astype(np.float32)
        self.X_prop[:] = np.nan

        # Start to store knockoffs
        self.Xk = np.zeros((self.n, self.p)).astype(np.float32)
        self.Xk[:] = np.nan

        # Decide whether or not to log
        if self.metro_verbose:
            print(f"Metro beginning to compute proposals...")
            j_iter = tqdm(range(self.p))
        else:
            j_iter = range(self.p)
        # Loop across variables to sample proposals
        for j in j_iter:

            # Sample proposal
            self.X_prop[:, j] = self.sample_proposals(
                X=self.X, prev_proposals=self.X_prop[:, 0:j]
            ).astype(np.float32)

        # Cache the conditional proposal params
        self.cache_conditional_proposal_params(
            verbose=self.metro_verbose, expensive_cache=self.cache
        )

        # Loop across variables to compute acc ratios
        prev_proposals = None
        if self.metro_verbose:
            print(f"Metro computing acceptance probabilities...")
            j_iter = tqdm(range(self.p))
        else:
            j_iter = range(self.p)
        for j in j_iter:

            # Cache which knockoff we are sampling
            self.step = j

            # Compute acceptance probability, which is an n-length vector
            acc_prob = self.compute_acc_prob(x_flags=np.zeros((self.n, self.p)), j=j,)
            self.final_acc_probs[:, j] = acc_prob

            # Sample to get actual acceptances
            self.acceptances[:, j] = np.random.binomial(1, acc_prob).astype(np.bool)

            # Store knockoffs
            mask = self.acceptances[:, j] == 1
            self.Xk[:, j][mask] = self.X_prop[:, j][mask]
            self.Xk[:, j][~mask] = self.X[:, j][~mask]

        # Delete cache to save memory
        if self.cache:
            if self.metro_verbose:
                print("Deleting cache to save memory...")
            del self.cached_mean_obs_eq_obs
            del self.cached_mean_obs_eq_prop

        # Return re-sorted
        return self.Xk[:, self.inv_order]


### Knockoff Samplers for T-distributions
def t_markov_loglike(X, rhos, df_t=3):
    """
    Calculates log-likelihood for markov chain
    specified in https://arxiv.org/pdf/1903.00434.pdf
    """
    p = X.shape[1]
    if rhos.shape[0] != p - 1:
        raise ValueError(
            f"Shape of rhos {rhos.shape} does not match shape of X {X.shape}"
        )
    inv_scale = np.sqrt(df_t / (df_t - 2))

    # Initial log-like for first variable
    loglike = t_log_likelihood(inv_scale * X[:, 0], df_t=df_t)

    # Differences: these are i.i.d. t
    # print(inv_scale * (X[:, 1:] - rhos * X[:, :-1]))
    conjugates = np.sqrt(1 - rhos ** 2)
    Zjs = inv_scale * (X[:, 1:] - rhos * X[:, :-1]) / conjugates
    Zj_loglike = t_log_likelihood(Zjs, df_t=df_t)

    # Add log-likelihood for differences
    return loglike + Zj_loglike.sum(axis=1)


class ARTKSampler(MetropolizedKnockoffSampler):
    """
    Samples knockoffs for autoregressive T-distributed designs.
    (Hence, ARTK). See https://arxiv.org/pdf/1903.00434.pdf
    for details.

    Parameters
    ----------
    X : np.ndarray
        the ``(n, p)``-shaped design matrix
    Sigma : np.ndarray
        ``(p, p)``-shaped covariance matrix of the features. The first
        diagonal should be the pairwise correlations which define the
        Markov chain.
    df_t : float
        The degrees of freedom for the t-distributions.
    kwargs : dict
        kwargs to pass to the constructor method of the generic
        ``MetropolizedKnockoffSampler`` class.
    """
    def __init__(self, X, Sigma, df_t, **kwargs):

        # Rhos and graph
        V = Sigma
        p = V.shape[0]
        self.df_t = df_t
        self.rhos = np.diag(V, 1)
        Q = utilities.chol2inv(V)

        # Cliques and clique log-potentials - start
        # with initial clique. Note that a log-potential
        # for a clique of size k takes an array of size
        # n x k as an input.
        cliques = [[0]]
        log_potentials = []
        inv_scale = np.sqrt(df_t / (df_t - 2))
        log_potentials.append(lambda X0: t_log_likelihood(inv_scale * X0, df_t=df_t))

        # Pairwise log-potentials
        def make_t_logpotential(rho, conj, invscale, df_t):
            def lp(Xc):
                return t_log_likelihood(
                    inv_scale * (Xc[:, 1] - rho * Xc[:, 0]) / conj, df_t=df_t
                )

            return lp

        conjugates = np.sqrt(1 - self.rhos ** 2)
        for i, rho, conj in zip(list(range(1, p)), self.rhos, conjugates):
            # Append the clique: X[:, [i+1,i]]
            cliques.append([i - 1, i])
            # Takes an n x 2 array as an input
            log_potentials.append(make_t_logpotential(rho, conj, inv_scale, df_t))

        # Loss function (unordered)
        def lf(X):
            return t_markov_loglike(X, self.rhos, self.df_t)

        super().__init__(
            lf=lf,
            X=X,
            mu=np.zeros(p),
            Sigma=V,
            undir_graph=np.abs(Q) > 1e-4,
            cliques=cliques,
            log_potentials=log_potentials,
            **kwargs,
        )


def t_mvn_loglike(X, invScale, mu=None, df_t=3):
    """
    Calculates multivariate t log-likelihood
    up to normalizing constant.
    :param X: n x p array of data
    :param invScale: p x p array, inverse multivariate t scale matrix
    :param mu: p-length array, location parameter
    :param df_t: degrees of freedom
    """
    p = invScale.shape[0]
    if mu is not None:
        X = X - mu
    quad_form = (np.dot(X, invScale) * X).sum(axis=1)
    log_quad = np.log(1 + quad_form / df_t)
    exponent = -1 * (df_t + p) / 2
    return exponent * log_quad


class BlockTSampler(KnockoffSampler):
    def __init__(self, X, Sigma, df_t, **kwargs):
        """
        Samples knockoffs for block multivariate t designs, where
        each block is independent.

        Parameters
        ----------
        X : np.ndarray
            the ``(n, p)``-shaped design matrix
        Sigma : np.ndarray
            ``(p, p)``-shaped covariance matrix of the features. This must
            be in block-diagonal form.
        df_t : float
            The degrees of freedom for the t-distributions.
        kwargs : dict
            kwargs to pass to the constructor method of the generic
            ``MetropolizedKnockoffSampler`` class.

        Attributes
        ----------
        samplers : list
            A list of ``MetropolizedKnockoffSampler`` objects used to sample
            knockoffs for each block.

        Notes
        -----
        Unlike the attributes of a ``MetropolizedKnockoffSampler`` class,
        the attributes of a ``BlockTSampler`` class are stored in the same
        order that the design matrix is initially passed in. E.g., ``self.Xk``
        corresponds with ``self.X``.
        """

        # Discover "block structure" of T
        V = Sigma
        self.p = V.shape[0]
        self.blocks, self.block_inds = dgp.cov2blocks(V)
        self.df_t = df_t
        self.X = X

        # Dummy order / inv_order variables for consistency
        self.order = np.arange(self.p)
        self.inv_order = np.arange(self.p)

        # Loop through blocks and initialize samplers
        self.samplers = []
        self.S = []
        for block, inds in zip(self.blocks, self.block_inds):

            # Invert block and create scale matrix
            inv_block = utilities.chol2inv(block)
            invScale = (df_t) / (df_t - 2) * inv_block

            # Undir graph is all connected
            blocksize = block.shape[0]
            undir_graph = np.ones((blocksize, blocksize))

            # Initialize sampler
            block_sampler = MetropolizedKnockoffSampler(
                lf=lambda X: t_mvn_loglike(X, invScale, df_t=df_t),
                X=X[:, inds],
                mu=np.zeros(blocksize),
                Sigma=block,
                undir_graph=undir_graph,
                **kwargs,
            )
            inv_order = block_sampler.inv_order
            self.S.append(block_sampler.S[:, inv_order][inv_order])
            self.samplers.append(block_sampler)

        # Concatenate S
        self.S = sp.linalg.block_diag(*self.S)

    def fetch_S(self):
        return self.S

    def sample_knockoffs(self, **kwargs):
        """
        Actually samples knockoffs sequentially for each block.

        Parameters
        ----------
        kwargs : dict
            kwargs for the ``MetropolizedKnockoffSampler.sample_knockoffs``
            call for each block.

        Returns
        -------
        Xk : np.ndarray
            A ``(n, p)``-shaped knockoff matrix in the original order 
            the variables were passed in.
        """
        # Loop through blocks and sample
        self.Xk = []
        self.final_acc_probs = []
        self.acceptances = []

        for j in range(len(self.samplers)):
            # Sample knockoffs
            Xk_block = self.samplers[j].sample_knockoffs(**kwargs)
            self.Xk.append(Xk_block)

            # Save final_acc_probs, acceptances
            block_acc_probs = self.samplers[j].final_acc_probs[
                :, self.samplers[j].inv_order
            ]
            self.final_acc_probs.append(block_acc_probs)
            block_acc = self.samplers[j].acceptances[:, self.samplers[j].inv_order]
            self.acceptances.append(block_acc)

        # Concatenate + return
        self.Xk = np.concatenate(self.Xk, axis=1)
        self.final_acc_probs = np.concatenate(self.final_acc_probs, axis=1)
        self.acceptances = np.concatenate(self.acceptances, axis=1)
        return self.Xk


class GibbsGridSampler(KnockoffSampler):
    """
    Samples knockoffs for a discrete gibbs grid using the divide-and-conquer
    algorithm plus metropolized knockoff sampling.

    Parameters
    ----------
    X : np.ndarray
        the ``(n, p)``-shaped design matrix
    gibbs_graph : np.ndarray
        ``(p, p)``-shaped matrix specifying the distribution
        of the gibbs grid: see ``knockpy.dgp.sample_gibbs``.
        This must correspond to a grid-like undirected graphical
        model.
    Sigma : np.ndarray
        ``(p, p)``-shaped estimated covariance matrix of the data.
    max_width : int 
        The maximum treewidth to allow in the divide-and-conquer
        algorithm.

    Notes
    -----
    Unlike the attributes of a ``MetropolizedKnockoffSampler`` class,
    the attributes of a ``BlockTSampler`` class are stored in the same
    order that the design matrix is initially passed in. E.g., ``self.Xk``
    corresponds with ``self.X``.
    """
    def __init__(self, X, gibbs_graph, Sigma, Q=None, mu=None, max_width=6, **kwargs):

        # Infer bucketization and V
        V = Sigma
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.X = X
        self.V = V
        self.S = None
        self.gibbs_graph = gibbs_graph
        self.gridwidth = int(np.sqrt(self.p))
        if self.gridwidth ** 2 != self.p:
            raise ValueError(f"p {self.p} must be a square number for gibbs grid")
        self.buckets = np.unique(X)
        if mu is None:
            mu = X.mean(axis=0)

        # Div and conquer makes this nontrivial
        # (Note to self: maybe TODO?)
        if "S" in kwargs:
            kwargs.pop("S")
        if "invSigma" in kwargs:
            kwargs.pop("invSigma")

        # Dummy order / inv_order variables for consistency
        self.order = np.arange(self.p)
        self.inv_order = np.arange(self.p)

        def make_ising_logpotential(temp, i, j):
            def lp(X):
                return -1 * temp * np.abs(X[:, 0] - X[:, 1])

            return lp

        # Learn cliques, log-potentials
        self.cliques = []
        self.log_potentials = []
        for i in range(self.p):
            for j in range(i):
                if gibbs_graph[i, j] != 0:
                    self.cliques.append((i, j))
                    temp = gibbs_graph[i, j]
                    self.log_potentials.append(make_ising_logpotential(temp, i, j))

        # Maps variables to cliques they're part of
        # Clique key maps the clique back to the log-potential
        self.clique_dict = {i: [] for i in range(self.p)}
        for clique_key, clique in enumerate(self.cliques):
            for j in clique:
                self.clique_dict[j].append((clique_key, clique))

        # Split X up into blocks along the n-axis.
        # Each divconq key corresponds to one way to
        # divide/conquer the variables, and to one
        # of these blocks.
        self.dc_keys = []
        for div_type in ["row", "col"]:
            for trans in [max_width - 1, max_width - 2]:
                self.dc_keys.append(f"{div_type}-{trans}")
        rand_inds = np.arange(self.n)
        np.random.shuffle(rand_inds)
        self.X_ninds = {key: [] for key in self.dc_keys}
        for i, j in enumerate(rand_inds):
            key = self.dc_keys[j % len(self.dc_keys)]
            self.X_ninds[key].append(i)

        # Structure of self.divconq_info
        # (1) Dictionary takes a divide-and-conquery key
        # (translation + row/col)
        # (2) This maps to a list of dictionaries. Each
        # dictionary corresponds to one of the blocks
        # in the divide and conquer procedure.
        # (3) Each dictionary takes three keys: inds,
        # cliques, lps.
        # - inds is the list of ORIGINAL coordinates
        # of the block of variables. E.g. if block 1
        # corresponds to columns 1,5,6, in X,
        # then inds = [1,5,6]
        # - cliques is a list of cliques in the NEW coordinates
        # of the block. So for example, if 1,5 was a clique in
        # the prior example, then (0,1) would be in this list.
        # - lps are the log-potentials corresponding to
        # the cliques above.
        self.divconq_info = {}

        # This maps divconq key
        # to a list of separators (e.g. knockoffs = features)
        # for these indices.
        self.separators = {}

        for dc_key in self.dc_keys:
            div_type = dc_key.split("-")[0]
            trans = int(dc_key.split("-")[-1])
            seps, dict_list = self._divide_variables(
                dc_key=dc_key,
                translation=trans,
                max_width=max_width,
                div_type=div_type,
            )
            self.separators[dc_key] = seps
            self.divconq_info[dc_key] = dict_list

        # Initialize samplers. Each sampler will only sample
        # a subset of variables for a subset of the data
        # Structure: maps divconq key to list of n_inds, p_inds, sampler
        self.samplers = {key: [] for key in self.dc_keys}
        for key in self.dc_keys:
            # Fetch indicies
            n_inds = np.array(self.X_ninds[key])
            sep_inds = self.separators[key]

            # Helper for conditional cov matrices
            V11 = V[sep_inds][:, sep_inds]  # s x s
            V11_inv = utilities.chol2inv(V11)

            for div_group_dict in self.divconq_info[key]:
                p_inds = div_group_dict["inds"]

                # Find conditional covariance matrix V
                # for p_inds given
                # the conditioned-on-separators
                V22 = V[p_inds][:, p_inds]  # p_i x p_i
                V21 = V[p_inds][:, sep_inds]  # p_i x s
                Vcond = V22 - np.dot(V21, np.dot(V11_inv, V21.T),)

                sampler = MetropolizedKnockoffSampler(
                    lf=None,
                    X=X[n_inds][:, p_inds],
                    mu=mu[p_inds],
                    Sigma=Vcond,
                    undir_graph=gibbs_graph[p_inds][:, p_inds] != 0,
                    cliques=div_group_dict["cliques"],
                    log_potentials=div_group_dict["lps"],
                    buckets=self.buckets,
                    S=None,
                    **kwargs,
                )
                if sampler.width > max_width:
                    warnings.warn(
                        f"Treewidth heuristic inaccurate during divide/conquer, sampler for {p_inds} has width {sampler.width} > max_width {max_width}"
                    )

                self.samplers[key].append((n_inds, p_inds, sampler))

    def num2coords(self, i):
        return dgp.num2coords(i=i, gridwidth=self.gridwidth)

    def coords2num(self, lc, wc):
        return dgp.coords2num(l=lc, w=wc, gridwidth=self.gridwidth)

    def _get_ac(self, i, div_type):
        """
        Helper function for divide-and-conquer
        in Ising model. Extracts active coordinate
        from variable i.
        Returns ac, lc, wc.
        """
        lc, wc = self.num2coords(i)
        if div_type == "row":
            return lc, lc, wc
        else:
            return wc, lc, wc

    def fetch_S(self):
        """ Returns ``None`` because the divide-and-conquer approach means
        there is no one S-matrix."""
        return None

    def sample_knockoffs(self, **kwargs):
        """
        Samples knockoffs using divide-and-conquer approach.

        Parameters
        ----------  
        kwargs : dict
            Keyword args for ``MetropolizedKnockoffSampler.sample_knockoffs``.

        Returns
        -------
        Xk : np.ndarray
            A ``(n, p)``-shaped knockoff matrix in the original order 
            the variables were passed in.
        """

        self.Xk = np.zeros((self.n, self.p))
        self.Xk[:] = np.nan
        self.final_acc_probs = self.Xk.copy()
        self.acceptances = self.Xk.copy()

        # Loop through different ways of separating variables
        for key in self.dc_keys:
            # N inds for this particular method of separation
            n_inds = np.array(self.X_ninds[key])

            # Initialize output
            Xkblock = np.zeros((len(n_inds), self.p))
            Xkblock[:] = np.nan
            accblock = Xkblock.copy()
            probblock = Xkblock.copy()

            # Set separating knockoffs = to their features
            sep_inds = self.separators[key]
            Xkblock[:, sep_inds] = self.X[n_inds][:, sep_inds]
            accblock[:, sep_inds] = 0
            probblock[:, sep_inds] = 0

            # Loop through blocks
            for n_inds, p_inds, sampler in self.samplers[key]:

                # Sample knockoffs
                Xkblock[:, p_inds] = sampler.sample_knockoffs(**kwargs)

                # Save final_acc_probs, acceptances
                accblock[:, p_inds] = sampler.final_acc_probs[:, sampler.inv_order]
                probblock[:, p_inds] = sampler.acceptances[:, sampler.inv_order]

            # Set Xk value for this block
            self.Xk[n_inds] = Xkblock
            self.acceptances[n_inds] = accblock
            self.final_acc_probs[n_inds] = probblock

        # Test validity
        self.check_xk_validity(self.X, self.Xk, testname="IsingSampler", alpha=1e-5)

        return self.Xk

    def _divide_variables(self, dc_key, translation, max_width, div_type):
        """
        Takes translation, max_width of junction tree, and div_type
        and returns separator_inds + a list of dictionaries. 
        dc_key is the divconq key.
        """
        # 0. Create separator variables
        separator_inds = [
            x for x in range(self.gridwidth) if x % max_width == translation
        ]
        separator_inds = np.array(separator_inds)
        separator_vars = []
        for s in separator_inds:
            for j in range(self.gridwidth):
                if div_type == "row":
                    separator_vars.append(self.coords2num(s, j))
                else:
                    separator_vars.append(self.coords2num(j, s))

        # 1. Use dgp.num2coords to determine the blocks
        # that each set of variables appear in (e.g.,
        # one block is columns 6-10, etc)
        div_groups = [[] for _ in range(len(separator_inds) + 1)]
        for i in range(self.p):
            # Determine the AC or active coordinate based on
            # whether or not we are splitting up rows / columns
            ac, _, _ = self._get_ac(i, div_type=div_type)

            # Ignore this variable if it is a separator
            if ac in separator_inds:
                continue

            # Groups to the right of lc
            right_groups = np.where(ac < separator_inds)[0]
            if right_groups.shape[0] == 0:
                div_groups[-1].append(i)
            else:
                div_groups[np.min(right_groups)].append(i)

        # Remove empty groups
        div_groups = [x for x in div_groups if len(x) > 0]

        # Quality check
        num_sep = len(set(separator_vars))
        nonsep = reduce(lambda x, y: x + y, div_groups)
        if num_sep + len(nonsep) != self.p:
            raise ValueError(
                f"Vars do not add to 1 ({num_sep} separators, {len(nonsep)} non-seps, p={self.p})"
            )

        # 2. Each block is now conditionally indepenent --
        # see proposition 3 of the paper. We need to change
        # the cliques to excise the "conditioned-on" variables.
        # Same thing holds for log-potentials
        def construct_trunc_logpotent(lp, j2, dc_key):
            def trunc_lp(Xj1):
                n_inds = self.X_ninds[dc_key]
                if Xj1.shape[0] != len(n_inds):
                    raise ValueError(
                        f"Xj1 shape {Xj1.shape} does not match num n_inds ({len(n_inds)}) for dc_key={dc_key}"
                    )
                Xc = np.stack([Xj1.reshape(-1), self.X[n_inds][:, j2]], axis=1)
                return lp(Xc)

            return trunc_lp

        output = []
        for div_group in div_groups:
            output.append({"inds": div_group})
            div_cliques = []
            div_lps = []
            for new_coord_j1, j1 in enumerate(div_group):
                # Original cliques for item j
                cliques_j = self.clique_dict[j1]
                for clique_key, clique in cliques_j:
                    # Find out which clique member is new
                    ind = 0 if clique[0] == j1 else 1
                    j2 = clique[1 - ind]
                    # Coordinates for analysis
                    ac1, lc1, wc1 = self._get_ac(j1, div_type)
                    ac2, lc2, wc2 = self._get_ac(j2, div_type)
                    # Check if the new clique size is 1 after conditioning
                    if j2 not in div_group:
                        # This means j2 must be separating the groups
                        if j2 not in separator_vars:
                            raise ValueError(
                                f"For j1={j1} (ac={ac1}, lc={lc1}, wc={lc2}), j2={j2} (ac={ac2}, lc={lc2}, wc={wc2}), div_group={div_group}, separators={separator_vars}, j2 not a separator"
                            )
                        # Replace Xj2 with X[:, j2] in log-potential
                        div_cliques.append([new_coord_j1])
                        # This only works because the log potentials are symmetric
                        # otherwise we'd have to condition on the ind
                        div_lps.append(
                            construct_trunc_logpotent(
                                lp=self.log_potentials[clique_key],
                                dc_key=dc_key,
                                j2=j2,
                            )
                        )
                    # Alternatively, for cliquesize = 2.
                    # We only append if j1 < j2, to prevent
                    # double-counting
                    elif j1 >= j2:
                        # Find the new coordinate for j2
                        new_coord_j2 = np.where(j2 == np.array(div_group))[0][0]
                        div_cliques.append([new_coord_j1, new_coord_j2])
                        div_lps.append(self.log_potentials[clique_key])

            # Add cliques + lps to output
            output[-1]["cliques"] = div_cliques
            output[-1]["lps"] = div_lps

        return separator_vars, output
