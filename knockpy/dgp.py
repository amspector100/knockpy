""" A collection of functions for generating synthetic datasets."""
import numpy as np
from scipy import stats

# Utility functions
from . import utilities
from .utilities import shift_until_PSD, chol2inv, cov2corr

# Tree methods
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as ssd

DEFAULT_DF_T = 3


def Wishart(d=100, p=100, tol=1e-2):
    """
    Let W be a random `d` x `p` matrix with i.i.d. Gaussian
    entries. Then Sigma = ``cov2corr``(W^T W).
    """

    W = np.random.randn(d, p)
    V = np.dot(W.T, W)
    V = cov2corr(V)
    return cov2corr(shift_until_PSD(V, tol=tol))


def UniformDot(d=100, p=100, tol=1e-2):
    """
    Let U be a random `d` x `p` matrix with i.i.d. uniform
    entries. Then Sigma = ``cov2corr``(U^T U)
    """
    U = np.random.uniform(size=(d, p))
    V = np.dot(U.T, U)
    V = cov2corr(V)
    return cov2corr(shift_until_PSD(V, tol=tol))


def DirichletCorr(p=100, temp=1, tol=1e-6):
    """
    Generates a correlation matrix by sampling p eigenvalues
    from a dirichlet distribution whose `p` parameters are i.i.d.
    uniform [`tol`, `temp`] and generating a random covariance matrix
    with those eigenvalues.
    """
    alpha = np.random.uniform(temp, size=p)
    d = stats.dirichlet(alpha=alpha).rvs().reshape(-1)

    # We have to round to prevent errors from random_correlation,
    # which is supperr sensitive to d.sum() != p even when the
    # error is a floating point error.
    d = np.around(d + tol, 6)
    d = p * d / d.sum()
    d[0] += p - d.sum()  # This is like 1e-10 but otherwise throws an error

    # Create and return matrix
    V = stats.random_correlation.rvs(d)
    return V


def AR1(p=30, a=1, b=1, tol=1e-3, max_corr=1, rho=None):
    """
    Generates `p`-dimensional correlation matrix for
    AR(1) Gaussian process, where successive correlations
    are drawn from Beta(`a`,`b`) independelty. If `rho` is
    specified, then the process is stationary with correlation
    `rho`.
    """

    # Generate rhos, take log to make multiplication easier
    if rho is None:
        rhos = np.log(np.minimum(
            stats.beta.rvs(size=p, a=a, b=b), max_corr
        ))
    else:
        if np.abs(rho) > 1:
            raise ValueError(f"rho {rho} must be a correlation between -1 and 1")
        rhos = np.log(np.array([rho for _ in range(p)]))
    rhos[0] = 0

    # Log correlations between x_1 and x_i for each i
    cumrhos = np.cumsum(rhos).reshape(p, 1)

    # Use cumsum tricks to calculate all correlations
    log_corrs = -1 * np.abs(cumrhos - cumrhos.transpose())
    corr_matrix = np.exp(log_corrs)

    # Ensure PSD-ness
    corr_matrix = cov2corr(shift_until_PSD(corr_matrix, tol))

    return corr_matrix


def NestedAR1(p=500, a=7, b=1, tol=1e-3, num_nests=5, nest_size=2):
    """
    Generates correlation matrix for AR(1) Gaussian process
    with hierarchical correlation structure.
    """
    # First set of correlations
    rhos = np.log(stats.beta.rvs(size=p, a=a, b=b))
    rhos[0] = 0

    # Create nested structure
    for j in range(1, num_nests + 1):
        rho_samples = np.log(stats.beta.rvs(size=p, a=a, b=b))
        # These indexes will get smaller correlations
        nest_inds = np.array(
            [x for x in range(p) if x % (nest_size ** j) == int(nest_size / 2)]
        ).astype("int")
        mask = np.zeros(p)
        mask[nest_inds] = 1
        rhos += mask * rho_samples

    # Generate cov matrix
    rhos[0] = 0
    cumrhos = np.cumsum(rhos).reshape(p, 1)

    # Use cumsum tricks to calculate all correlations
    log_corrs = -1 * np.abs(cumrhos - cumrhos.transpose())
    corr_matrix = np.exp(log_corrs)

    # Ensure PSD-ness
    corr_matrix = cov2corr(shift_until_PSD(corr_matrix, 1e-3))

    return corr_matrix


def ErdosRenyi(p=300, delta=0.2, lower=0.1, upper=1, tol=1e-1):
    """ 
    Randomly samples bernoulli flags as well as values
    for partial correlations to generate sparse square
    matrices. Follows https://arxiv.org/pdf/1908.11611.pdf.
    """

    # Initialization
    V = np.zeros((p, p))
    triang_size = int((p ** 2 + p) / 2)

    # Sample the upper triangle
    mask = stats.bernoulli.rvs(delta, size=triang_size)
    vals = np.random.uniform(lower, upper, size=triang_size)
    flags = 2 * np.random.binomial(1, 0.5, triang_size) - 1
    triang = mask * flags * vals

    # Set values and add diagonal
    upper_inds = np.triu_indices(p, 0)
    V[upper_inds] = triang
    V = V + V.T
    V += np.eye(p) - np.diag(np.diag(V))

    # Force to be positive definite -
    V = shift_until_PSD(V, tol=tol)

    return V


def FactorModel(
    p=500, rank=2,
):
    """
    Generates random correlation matrix from a factor model 
    with dimension `p` and rank `rank`.
    """
    diag_entries = np.random.uniform(low=0.01, high=1, size=p)
    noise = np.random.randn(p, rank) / np.sqrt(rank)
    V = np.diag(diag_entries) + np.dot(noise, noise.T)
    V = utilities.cov2corr(V)
    Q = utilities.chol2inv(V)
    return V, Q


def PartialCorr(
    p=300, rho=0.3,
):
    """
    Creates a correlation matrix of dimension `p` with partial correlation `rho`.
    """
    Q_init = rho * np.ones((p, p)) + (1 - rho) * np.eye(p)
    V = utilities.chol2inv(Q_init)
    V = utilities.cov2corr(V)
    Q = utilities.chol2inv(V)
    return V, Q


def block_equi_graph(
    n=3000,
    p=1000,
    block_size=5,
    sparsity=0.1,
    rho=0.5,
    gamma=0,
    coeff_size=3.5,
    coeff_dist=None,
    sign_prob=0.5,
    iid_signs=True,
    corr_signals=False,
    beta=None,
    mu=None,
    **kwargs,
):
    """ 
    Samples data according to a block-equicorrelated Gaussian design,
    where ``rho`` is the within-block correlation and ``gamma * rho``
    is the between-block correlation.
    
    Parameters
    ----------
    n : int
        The number of data points
    p : int
        The dimensionality of the data
    block_size : int
        The size of blocks. Defaults to 5.
    sparsity : float
        The proportion of groups which are null. Defaults to 0.1
    rho : float
        The within-group correlation
    gamma : float
        The between-group correlation is ``gamma * rho``
    beta : np.ndarray
        If supplied, the ``(p,)``-shaped set of coefficients. Else,
        will be generated by calling 
        ``knockpy.dgp.sample_sparse_coefficients.``
    mu : np.ndarray
        The ``p``-dimensional mean of the covariates. Defaults to 0.
    kwargs : dict
        Args passed to ``knockpy.dgp.sample_response``.

    Notes
    -----
    This defaults to the same data-generating process as
    Dai and Barber 2016 (see https://arxiv.org/abs/1602.03589).
    """

    # Set default values
    num_groups = int(p / block_size)

    # Create blocks / blocks
    groups = np.array([int(i / (p / num_groups)) for i in range(p)])

    # Helper fn for covariance matrix
    def get_corr(g1, g2):
        if g1 == g2:
            return rho
        else:
            return gamma * rho

    get_corr = np.vectorize(get_corr)

    # Create correlation matrix, invert
    Xcoords, Ycoords = np.meshgrid(groups, groups)
    Sigma = get_corr(Xcoords, Ycoords)
    Sigma = Sigma + np.eye(p) - np.diagflat(np.diag(Sigma))
    Q = chol2inv(Sigma)

    # Create beta
    if beta is None:
        beta = create_sparse_coefficients(
            p=p,
            sparsity=sparsity,
            coeff_size=coeff_size,
            groups=groups,
            sign_prob=sign_prob,
            iid_signs=iid_signs,
            coeff_dist=coeff_dist,
            corr_signals=corr_signals,
            n=n,
        )

    # Sample design matrix
    if mu is None:
        mu = np.zeros(p)
    X = stats.multivariate_normal.rvs(mean=mu, cov=Sigma, size=n)
    # Sample y
    y = sample_response(X, beta, **kwargs)

    return X, y, beta, Q, Sigma, groups + 1


def create_sparse_coefficients(
    p,
    sparsity=0.5,
    groups=None,
    coeff_size=1,
    coeff_dist=None,
    sign_prob=0.5,
    iid_signs=True,
    corr_signals=False,
    n=None,
):
    """
    Generate a set of sparse coefficients for single index or sparse additive models.
    p : int
        Dimensionality of coefficients 
    sparsity : float
        Proportion of non-null coefficients. Generates ``np.floor(p*sparsity)`` non-nulls.
    coeff_size: float
        Size of non-zero coefficients
    groups : np.ndarray
        A p-length array of integers from 1 to num_groups such that
        ``groups[j] == i`` indicates that variable `j` is a member of group `i`.
        Defaults to ``None``. Else, floor(sparsity * num_groups) groups will be 
        chosen to be non-null, where all elements of each group are non-null.
    sign_prob : float
        The probability that each nonzero coefficient will be positive. 
    iid_signs : bool
        If True, the signs of the coeffs are assigned independently.
        Else, exactly sign_prob*sparsity*p coefficients will be positive.
    coeff_dist : str
        Specifies the distribution of nonzero coefficients. Three options:
        - ``None``: all coefficients have absolute value coeff_size.
        - ``normal``: all nonzero coefficients are drawn from Normal(coeff_size, 1). 
        - ``uniform``: nonzero coeffs are drawn from Unif(coeff_size/2, coeff_size).
    corr_signals : bool
        If true, all of the nonzero coefficients will lie in a consecutive block.

    Returns
    -------
    beta : np.ndarray
        ``(p,)``-shaped array of sparse coefficients
    """

    # First, decide which coefficients are nonzero, one of two methods
    if groups is not None:

        # Method one: a certain number of groups are nonzero
        num_groups = np.unique(groups).shape[0]
        num_nonzero_groups = int(np.floor(sparsity * num_groups))
        chosen_groups = np.random.choice(
            np.unique(groups), num_nonzero_groups, replace=False
        )
        beta = np.array([coeff_size if i in chosen_groups else 0 for i in groups])

    else:

        # Method two (default): a certain percentage of coefficients are nonzero
        num_nonzero = int(np.floor(sparsity * p))
        if corr_signals:
            nonnull_start = np.random.randint(0, p - num_nonzero + 1)
            beta = np.zeros(p)
            beta[nonnull_start : nonnull_start + num_nonzero] = coeff_size
        else:
            beta = np.array([coeff_size] * num_nonzero + [0] * (p - num_nonzero))
            np.random.shuffle(beta)

    beta_nonzeros = beta != 0
    num_nonzero = beta_nonzeros.sum()

    # Now draw random signs
    if iid_signs:
        signs = 1 - 2 * stats.bernoulli.rvs(sign_prob, size=p)
        beta = beta * signs
    else:
        num_pos = int(np.floor(num_nonzero * sign_prob))
        signs = np.concatenate(
            [np.ones((num_pos)), -1 * np.ones((num_nonzero - num_pos))]
        )
        np.random.shuffle(signs)
        beta[beta_nonzeros] = beta[beta_nonzeros] * signs

    # Possibly change the absolute values of beta
    if coeff_dist is not None:
        if str(coeff_dist).lower() == "normal":
            beta = (np.sqrt(coeff_size) * np.random.randn(p)) * beta_nonzeros
        elif str(coeff_dist).lower() == "uniform":
            beta = beta * np.random.uniform(size=p) / 2 + beta / 2
        elif str(coeff_dist).lower() == "dsliu2020":
            if num_nonzero != 50:
                raise ValueError(
                    f"To replicate dsliu2020 paper, need num_nonzero ({num_nonzero})=50"
                )
            variance = 10 * np.sqrt(np.log(p) / n)
            beta = beta_nonzeros * np.sqrt(variance) * np.random.randn(p)
        elif str(coeff_dist).lower() == "gmliu2019":
            if num_nonzero != 60:
                raise ValueError(
                    f"To replicate gmliu2019 paper, need num_nonzero ({num_nonzero})=60"
                )
            variance = 20 / np.sqrt(n)
            beta = beta_nonzeros * np.sqrt(variance) * np.random.randn(p)
        elif str(coeff_dist).lower() == "none":
            pass
        else:
            raise ValueError(
                f"coeff_dist ({coeff_dist}) must be 'none', 'normal', or 'uniform'"
            )

    return beta


def sample_response(X, beta, cond_mean="linear", y_dist="gaussian"):
    """ 
    Given a design matrix X and coefficients beta, samples a response y.

    X : np.ndarray
        ``(n, p)``-shaped design matrix
    beta : np.ndarray
        ``(p, )``-shaped coefficient vector
    cond_mean : str
        How to calculate the conditional mean of y given X, denoted mu(X). 
        Six options:
            1. "linear" denotes ``np.dot(X, beta)``
            2. "cubic" denotes ``np.dot(X**3, beta) - np.dot(X, beta)``
            3. "trunclinear" ``((X * beta >= 1).sum(axis = 1))``
            Stands for truncated linear.
            4. "pairint": pairs up non-null coefficients according to the
            order of beta, multiplies them and their beta values, then
            sums. "pairint" stands for pairwise-interactions.
            5. "cos": ``mu(X) = sign(beta) * (beta != 0) * np.cos(X)``
            6. "quadratic": ``mu(X) = np.dot(np.power(X, 2), beta)``
    y_dist : str
        If "gaussian", y is the conditional mean plus gaussian noise.
        If "binomial", Pr(y=1) = softmax(cond_mean).

    Returns
    -------
    y : np.ndarray
        ``(n,)``-shaped response vector
    """

    n = X.shape[0]
    p = X.shape[1]

    if cond_mean == "linear":
        cond_mean = np.dot(X, beta)
    elif cond_mean == "quadratic":
        cond_mean = np.dot(np.power(X, 2), beta)
    elif cond_mean == "cubic":
        cond_mean = np.dot(np.power(X, 3), beta) - np.dot(X, beta)
    elif cond_mean == "trunclinear":
        cond_mean = ((X * beta >= 1) * np.sign(beta)).sum(axis=1)
    elif cond_mean == "cos":
        cond_mean = (np.sign(beta) * (beta != 0) * np.cos(X)).sum(axis=1)
    elif cond_mean == "pairint":
        # Pair up the coefficients
        pairs = [[]]
        for j in range(p):
            if beta[j] != 0:
                if len(pairs[-1]) < 2:
                    pairs[-1].append(j)
                else:
                    pairs.append([j])

        # Multiply pairs and add to cond_mean
        cond_mean = np.zeros(n)
        for pair in pairs:
            interaction_term = np.ones(n)
            for j in pair:
                interaction_term = interaction_term * beta[j]
                interaction_term = interaction_term * X[:, j]
            cond_mean += interaction_term
    else:
        raise ValueError(
            f"cond_mean must be one of 'linear', 'quadratic', cubic', 'trunclinear', 'cos', 'pairint', not {cond_mean}"
        )

    # Create y, one of two families
    if y_dist == "gaussian":
        y = cond_mean + np.random.standard_normal((n))
    elif y_dist == "binomial":
        probs = 1 / (1 + np.exp(-1 * cond_mean))
        y = stats.bernoulli.rvs(probs)
    else:
        raise ValueError(f"y_dist must be one of 'gaussian', 'binomial', not {y_dist}")

    return y


def sample_ar1t(
    rhos, n=50, df_t=DEFAULT_DF_T,
):
    """
    Samples t-distributed variables according to a Markov chain.
    
    Parameters
    ----------
    rhos : np.ndarray
        ``(p-1, )``-length array of correlations between consecutive
        variables.
    n : int
        The number of data points to sample
    df_t : float
        The degrees of freedom of the t variables

    Returns
    -------
    X : np.ndarray
        ``(n, p)``-shaped design matrix
    """
    # Initial t samples
    p = rhos.shape[0] + 1
    tvars = stats.t(df=df_t).rvs(size=(n, p))

    # Initialize X
    X = np.zeros((n, p))
    scale = np.sqrt((df_t - 2) / df_t)
    X[:, 0] = scale * tvars[:, 0]

    # Loop through variables according to markov chain
    conjugates = np.sqrt(1 - rhos ** 2)
    for j in range(1, p):
        X[:, j] = rhos[j - 1] * X[:, j - 1] + conjugates[j - 1] * scale * tvars[:, j]

    return X


def cov2blocks(V, tol=1e-5):
    """
    Decomposes a PREORDERED block-diagonal matrix V into its blocks.
    """
    p = V.shape[0]
    blocks = []
    block_start = 0
    block_inds = []
    for j in range(p + 1):
        # Detect if we have exited the block
        if j == p:
            blocks.append(V[block_start:j, block_start:j])
            block_inds.append(list(range(block_start, j)))
        elif np.abs(V[block_start, j]) < tol:
            # If so, reset the block_start
            blocks.append(V[block_start:j, block_start:j])
            block_inds.append(list(range(block_start, j)))
            block_start = j

    return blocks, block_inds


def sample_block_tmvn(
    blocks, n=50, df_t=DEFAULT_DF_T,
):
    """
    Samples a blocks of multivariate-t distributed variables
    according to a list of covariance matrices called ``blocks``.

    Parameters
    ----------
    blocks : list
        A list of square, symmetric numpy arrays. These are
        the covariance matrices for each block of variables.
    n : int
        The number of data points to sample
    df_t : float
        The degrees of freedom of the t-distribution

    Notes
    -----
    The variables are scaled such that the marginal variance
    of each variable equals a diagonal element in one of the blocks.
    """

    # Possibly calculate sqrt roots
    #if block_sqrts is None:
    block_sqrts = [np.linalg.cholesky(block) for block in blocks]
    # if len(block_sqrts) != len(blocks):
    #     raise ValueError(f"Blocks and block_sqrts must have same length")

    # Loop through blocks and sample multivariate t
    X = []
    for i, block_sqrt in enumerate(block_sqrts):
        # Dimensionality and also sample chisquares
        p_block = block_sqrt.shape[0]
        chi_block = stats.chi2.rvs(df=df_t, size=(n, 1))

        # Linear transformatino + chi square multiplication
        Z_block = np.random.randn(n, p_block)  # n x p
        t_block = np.dot(Z_block, block_sqrt.T)  # n x p
        t_block = np.sqrt(df_t / chi_block) * t_block
        X.append(t_block)

    #  Append back together and scale such that variance is 1
    scale = np.sqrt((df_t - 2) / (df_t))
    X = scale * np.concatenate(X, axis=1)
    return X


### Helper functions for Gibbs Sampling
def num2coords(i, gridwidth=10):
    """
    Coordinates of variable i in a Gibbs grid
    
    Parameters
    ----------
    i : int
        Position of variable in ordering
    gridwidth : int
        Width of the grid

    Returns
    -------
    length_coord, width_coord (coordinates)
    """
    length_coord = i % gridwidth
    width_coord = i // gridwidth
    return int(length_coord), int(width_coord)


def coords2num(l, w, gridwidth=10):
    """ Takes coordinates of variable in a Gibbs grid, returns position"""
    if l < 0 or w < 0:
        return -1
    if l >= gridwidth or w >= gridwidth:
        return -1
    return int(w * gridwidth + l)


def graph2cliques(Q):
    """
    Turns graph Q of connections into binary cliques for Gibbs grid
    """
    p = Q.shape[0]
    clique_dict = {i: [] for i in range(p)}
    for i in range(p):
        for j in range(i):
            if Q[i, j] != 0:
                clique_dict[i].append((i, j))
                clique_dict[j].append((j, i))
    # Remove duplicates
    for i in range(p):
        clique_dict[i] = list(set(clique_dict[i]))
    return clique_dict

def construct_gibbs_grid(n, p, temp=1):
    """
    Creates gridlike ``gibbs_graph`` parameter for ``sample_gibbs``.
    See ``sample_gibbs``.
    """
    # Infer dimensionality
    gridwidth = int(np.sqrt(p))
    variables = set(list(range(p)))

    # Generate
    gibbs_graph = np.zeros((p, p))  
    for i1 in range(p):
        lc, wc = num2coords(i1, gridwidth=gridwidth)
        for ladd in [-1, 1]:
            i2 = coords2num(lc + ladd, wc, gridwidth=gridwidth)
            if i2 != -1:
                sign = 1 - 2 * np.random.binomial(1, 0.5)
                gibbs_graph[i1, i2] = temp * sign
                gibbs_graph[i2, i1] = temp * sign
        for wadd in [-1, 1]:
            i2 = coords2num(lc, wc + wadd, gridwidth=gridwidth)
            if i2 != -1:
                sign = 1 - 2 * np.random.binomial(1, 0.5)
                gibbs_graph[i1, i2] = temp * sign
                gibbs_graph[i2, i1] = temp * sign

    return gibbs_graph

def sample_gibbs(
    n, p, gibbs_graph=None, temp=1, num_iter=15, K=20, max_val=2.5,
):
    """ Samples from a discrete Gibbs measure using a gibbs sampler.

    The joint likelihood for the `p`-dimensional data X1 through Xp
    is the product of all terms of the following form: 
        ```np.exp(-1*gibbs_graph[i,j]*np.abs(Xi - Xj))```
    where `gibbs_graph` is assumed to be symmetric (and is usually
    sparse). Each feature takes values on an evenly spaced grid from
    ``-1*max_val`` to ``max_val`` with ``K`` values.

    Parameters
    ----------
    n : int
        How many data points to sample
    p : int
        Dimensionality of the data
    gibbs_graph : np.ndarray
        A symmetric ``(p, p)``-shaped array. See likelihood equation.
        By default, this is corresponds to an undirected graphical
        model with a square grid (like an Ising model) with nonzero
        entries set to 1 or -1 with equal probability.
    temp : float
        Governs the strength of interactions between features---see 
        the likelihood equation.
    num_iter : int
        Number of iterations in the Gibbs sampler; defaults to 15.
    K : int
        Number of discrete values each sampled feature can take.
    max_val : float
        The maximum absolute value each feature can take.

    Returns
    -------
    X : np.ndarray
        ``(n, p)``-shaped array of data
    gibbs_graph : np.ndarray
        The generated `gibbs_graph`.
    """
    # Create buckets from (approximately) -max_val to max_val
    buckets = np.arange(-K + 1, K + 1, 2) / (K / max_val)

    # Log potentials generator
    def log_potential(X1, X2=None, temp=1):
        if X2 is None:
            X2 = X1[:, 1]
            X1 = X1[:, 0]
        return -1 * temp * np.abs(X1 - X2)

    # Construct cliques from gibbs_graph
    if gibbs_graph is None:
        gibbs_graph = construct_gibbs_grid(n=n, p=p, temp=temp)
    clique_dict = graph2cliques(gibbs_graph)

    # Initialize
    X = np.random.randn(n, p, 1)
    dists = np.abs(X - buckets.reshape(1, 1, -1))
    indices = dists.argmin(axis=-1)
    X = buckets[indices]

    # Marginalize / gibbs sample
    for _ in range(num_iter):
        for j in range(p):
            # Cliques and marginals for this node
            cliques = clique_dict[j]
            marginals = np.zeros((n, K))
            for clique in cliques:
                # Calculate log-potential from this clique
                X1 = buckets.reshape(1, -1)
                X2 = X[:, clique[-1]].reshape(-1, 1)
                marginals += log_potential(
                    X1=X1, X2=X2, temp=gibbs_graph[j, clique[-1]]
                )

            # Resample distribution of this value of X
            # Note the exp is a bottleneck here for efficiency
            marginals = np.exp(marginals.astype(np.float32))
            marginals = marginals / marginals.sum(axis=-1, keepdims=True)
            marginals = marginals.cumsum(axis=-1)

            # Batched multinomial sampling using uniforms
            # (faster than for loops + numpy)
            unifs = np.random.uniform(size=(n, 1))
            Xnew = buckets[np.argmax(unifs <= marginals, axis=-1)]
            X[:, j] = Xnew

    return X, gibbs_graph


class DGP:
    """
    A utility class which creates a (random) data-generating process 
    for a design matrix X and a response y. If the parameters are
    not specified, they will be randomly generated during the
    ``sample_data`` method. 

    Parameters
    ----------
    mu : np.ndarray
        ``(p,)``-shaped mean vector for X data
    Sigma : np.ndarray
        ``(p, p)``-shaped covariance matrix for X data
    invSigma : np.ndarray
        ``(p, p)``-shaped precision matrix for X data
    beta : np.ndarray
        coefficients used to generate y from X in a single 
        index or sparse additive model.
    gibbs_graph : np.ndarray
        ``(p, p)``-shaped matrix of coefficients for gibbs
        grid method.

    Attributes
    ----------
    mu : np.ndarray
        See above
    Sigma : np.ndarray
        See above
    invSigma : np.ndarray
        See above
    beta : np.ndarray
        See above
    gibbs_Graph : np.ndarray
        See above
    X : np.ndarray
        ``(n, p)``-shaped design matrix
    y : np.ndarray
        ``(n, )``-shaped response vector
    """

    def __init__(
            self, 
            mu=None,
            Sigma=None, 
            invSigma=None,
            beta=None,
            gibbs_graph=None
        ):

        self.mu = mu
        self.Sigma = Sigma
        self.invSigma = invSigma
        self.beta = beta
        self.gibbs_graph = gibbs_graph

    def sample_data(
        self,
        p=100,
        n=50,
        x_dist="gaussian",
        method="AR1",
        y_dist="gaussian",
        cond_mean="linear",
        coeff_size=1,
        coeff_dist=None,
        sparsity=0.5,
        groups=None,
        sign_prob=0.5,
        iid_signs=True,
        corr_signals=False,
        df_t=DEFAULT_DF_T,
        **kwargs,
    ):
        """
        (Possibly) generates random data-generating parameters
        and then samples data using those parameters. By default,
        (X, y) are jointly Gaussian and y is a linear response to
        X.

        Parameters
        ----------
        n : int
            The number of data points
        p : int
            The dimensionality of the data
        x_dist : str
            Specifies the distribution of X. One of "Gaussian", "blockt", "ar1t", or "gibbs".
        method : str
            How to generate the covariance matrix of X. One of
            "AR1", "NestedAR1", "partialcorr", "factor", 
            "blockequi", "ver", "qer", "dirichlet", or "uniformdot".
            See the docs for each of these methods.
        y_dist : str
            Specifies the distribution of y. One of "Gaussian" or "binomial".
        cond_mean : str 
            How to calculate the conditional mean of y given X. Defaults to 
            "linear". See ``knockpy.dgp.sample_response`` for more options.
        coeff_size: float
            Size of non-zero coefficients
        coeff_dist : str
            Specifies the distribution of nonzero coefficients. Three options:
            - ``None``: all coefficients have absolute value coeff_size.
            - ``normal``: all nonzero coefficients are drawn from Normal(coeff_size, 1). 
            - ``uniform``: nonzero coeffs are drawn from Unif(coeff_size/2, coeff_size).
        sparsity : float
            Proportion of non-null coefficients. 
            Generates ``np.floor(p*sparsity)`` non-nulls.
        groups : np.ndarray
            A p-length array of integers from 1 to num_groups such that
            ``groups[j] == i`` indicates that variable `j` is a member of group `i`.
            Defaults to ``None``. Else, floor(sparsity * num_groups) groups will be 
            chosen to be non-null, where all elements of each group are non-null.
        sign_prob : float
            The probability that each nonzero coefficient will be positive. 
        iid_signs : bool
            If True, the signs of the coeffs are assigned independently.
            Else, exactly sign_prob*sparsity*p coefficients will be positive.
        corr_signals : bool
            If true, all of the nonzero coefficients will lie in a consecutive block.
        df_t : float
            If the X variables are marginally t-distributed, the degrees of freedom.
        kwargs: dict
            keyword arguments to pass to method for generating the covariance matrix.
        """

        # Defaults
        if self.mu is None:
            self.mu = np.zeros(p)

        # grid / Gibbs Sampling
        if x_dist == "gibbs":
            # Sample X, Q
            self.X, self.gibbs_graph = sample_gibbs(
                n=n, p=p, gibbs_graph=self.gibbs_graph, **kwargs
            )

            # Estimate cov matrix
            if self.Sigma is None:
                self.Sigma, _ = utilities.estimate_covariance(
                    self.X, tol=1e-6, shrinkage=None
                )
            self.invSigma = None

        # Create covariance matrix
        if self.invSigma is None and self.Sigma is None:

            method = str(method).lower()
            if method == "ar1":
                self.Sigma = AR1(p=p, **kwargs)
                self.invSigma = chol2inv(self.Sigma)
            elif method == "nestedar1":
                self.Sigma = NestedAR1(p=p, **kwargs)
                self.invSigma = chol2inv(self.Sigma)
            elif method == "partialcorr":
                self.Sigma, self.invSigma = PartialCorr(p=p, **kwargs)
            elif method == "factor":
                self.Sigma, self.invSigma = FactorModel(p=p, **kwargs)
            elif method == "equi" or method == "blockequi":
                _,_,self.beta,self.invSigma,self.Sigma,_ = block_equi_graph(
                    p=p,
                    n=n,
                    coeff_size=coeff_size,
                    coeff_dist=coeff_dist,
                    sparsity=sparsity,
                    sign_prob=sign_prob,
                    iid_signs=iid_signs,
                    corr_signals=corr_signals,
                    beta=self.beta,
                    **kwargs,
                )
            elif method == "ver":
                max_corr = kwargs.pop('max_corr', 1) # enforce maximum corr
                self.Sigma = cov2corr(ErdosRenyi(p=p, **kwargs))
                self.Sigma = np.maximum(np.minimum(self.Sigma, max_corr), -1*max_corr)
                self.Sigma += np.eye(p) - np.diag(np.diag(self.Sigma))
                self.invSigma = chol2inv(self.Sigma)
            elif method == "qer":
                self.invSigma = ErdosRenyi(p=p, **kwargs)
                self.Sigma = cov2corr(chol2inv(self.invSigma))
                self.invSigma = chol2inv(self.Sigma)
            elif method == "dirichlet":
                self.Sigma = DirichletCorr(p=p, **kwargs)
                self.invSigma = chol2inv(self.Sigma)
            elif method == "wishart":
                self.Sigma = Wishart(p=p, **kwargs)
                self.invSigma = chol2inv(self.Sigma)
            elif method == "uniformdot":
                self.Sigma = UniformDot(p=p, **kwargs)
                self.invSigma = chol2inv(self.Sigma)
            else:
                raise ValueError(f"Other method {method} not implemented yet")

        elif self.invSigma is None:
            self.invSigma = chol2inv(self.Sigma)
        elif self.Sigma is None:
            self.Sigma = cov2corr(chol2inv(self.invSigma))
            self.invSigma = chol2inv(self.Sigma)
        else:
            pass

        # Create sparse coefficients
        if self.beta is None:
            self.beta = create_sparse_coefficients(
                p=p,
                sparsity=sparsity,
                coeff_size=coeff_size,
                groups=groups,
                sign_prob=sign_prob,
                iid_signs=iid_signs,
                coeff_dist=coeff_dist,
                corr_signals=corr_signals,
                n=n,
            )

        # Sample design matrix
        if x_dist == "gibbs":
            pass
        elif x_dist == "gaussian":
            self.X = stats.multivariate_normal.rvs(mean=self.mu, cov=self.Sigma, size=n)
        elif x_dist == "ar1t":
            if str(method).lower() != "ar1":
                raise ValueError(
                    f"For x_dist={x_dist}, method ({method}) should equal 'ar1'"
                )
            self.X = sample_ar1t(n=n, rhos=np.diag(self.Sigma, 1), df_t=df_t)
        elif x_dist == "blockt":
            blocks, _ = cov2blocks(self.Sigma)
            self.X = sample_block_tmvn(blocks, n=n, df_t=df_t)
        else:
            raise ValueError(
                f"x_dist must be one of 'gaussian', 'gibbs', 'ar1t', 'blockt'"
            )

        # Sample y
        self.y = sample_response(
            X=self.X, beta=self.beta, y_dist=y_dist, cond_mean=cond_mean
        )
        return self.X, self.y, self.beta, self.invSigma, self.Sigma


def create_correlation_tree(corr_matrix, method="single"):
    """ Creates hierarchical clustering (correlation tree)
    from a correlation matrix

    Parameters
    ----------
    corr_matrix : np.ndarray
        ``(p, p)``-shaped correlation matrix
    method : str
        the method of hierarchical clustering:
        'single', 'average', 'fro', or 'complete'.
        Defaults to 'single'.
    
    Returns
    -------
    link : np.ndarray
        The `link` of the correlation tree, as in scipy
    """

    # Distance matrix for tree method
    if method == "fro":
        dist_matrix = np.around(1 - np.power(corr_matrix, 2), decimals=7)
    else:
        dist_matrix = np.around(1 - np.abs(corr_matrix), decimals=7)
    dist_matrix -= np.diagflat(np.diag(dist_matrix))

    condensed_dist_matrix = ssd.squareform(dist_matrix)

    # Create linkage
    if method == "single":
        link = hierarchy.single(condensed_dist_matrix)
    elif method == "average" or method == "fro":
        link = hierarchy.average(condensed_dist_matrix)
    elif method == "complete":
        link = hierarchy.complete(condensed_dist_matrix)
    else:
        raise ValueError(
            f'Only "single", "complete", "average", "fro" are valid methods, not {method}'
        )

    return link

def create_grouping(Sigma, cutoff=0.95, **kwargs):
    """ Creates a knockoff grouping from a covariance matrix
    using hierarchical clustering.

    Parameters
    ----------
    Sigma : np.ndarray
        ``(p, p)``-shaped covariance matrix
    cutoff : np.ndarray
        The correlation at which to "cut" the correlation tree.
        For method='single', cutoff=0.95 ensures that
        no two variables in different groups have absolute correlation
        greater than 0.95.
    
    Returns
    -------
    groups : np.ndarray
        `(p,)`-shaped array of groups for each variable
    """


    # Create correlation tree
    corr_matrix = cov2corr(Sigma)
    link = create_correlation_tree(
        corr_matrix, **kwargs
    )
    return hierarchy.fcluster(link, 1-cutoff, criterion="distance")