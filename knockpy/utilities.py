import warnings
import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigsh
import sklearn.covariance
import itertools
from multiprocessing import Pool
from functools import partial

### Group helpers
def preprocess_groups(groups):
    """
    Maps the m unique elements of a 1D "groups" array to the integers from 1 to m.
    """
    unique_vals = np.unique(groups)
    conversion = {unique_vals[i]: i for i in range(unique_vals.shape[0])}
    return np.array([conversion[x] + 1 for x in groups])


def fetch_group_nonnulls(non_nulls, groups):
    """ 
    Combines feature-level null hypotheses into group-level hypothesis.
    """

    if not isinstance(non_nulls, np.ndarray):
        non_nulls = np.array(non_nulls)
    if not isinstance(groups, np.ndarray):
        groups = np.array(groups)

    # Initialize
    m = np.unique(groups).shape[0]
    group_nonnulls = np.zeros(m)

    # Calculate and return
    for j in range(m):
        flag = np.abs(non_nulls[groups == j + 1]).sum() > 0
        group_nonnulls[j] = float(flag)
    return group_nonnulls


def calc_group_sizes(groups):
    """
    Given a list of groups, finds the sizes of the groups.
    
    Parameters
    ----------
    groups : np.ndarray
        ``(p, )``-shaped array which takes m integer values from
        1 to m. If ``groups[i] == j``, this indicates that coordinate
        ``i`` belongs to group ``j``.
    :param groups: p-length array of integers between 1 and m, 
    
    Returns
    -------
    sizes : np.ndarray
        ``(m, )``-length array of group sizes.
    """
    if not isinstance(groups, np.ndarray):
        groups = np.array(groups)
    if np.all(groups.astype("int32") != groups):
        raise TypeError(
            "groups cannot contain non-integer values: apply preprocess_groups first"
        )
    else:
        groups = groups.astype("int32")

    if np.min(groups) == 0:
        raise ValueError(
            "groups cannot contain 0: add one or apply preprocess_groups first"
        )

    m = groups.max()
    group_sizes = np.zeros(m)
    for j in groups:
        group_sizes[j - 1] += 1
    group_sizes = group_sizes.astype("int32")
    return group_sizes


### Matrix helpers for S-matrix computation
def cov2corr(M):
    """ Rescales a p x p cov. matrix M to be a correlation matrix """
    scale = np.sqrt(np.diag(M))
    return M / np.outer(scale, scale)


def chol2inv(X):
    """ Uses cholesky decomp to get inverse of matrix """
    triang = np.linalg.inv(np.linalg.cholesky(X))
    return np.dot(triang.T, triang)


def shift_until_PSD(M, tol):
    """ Add the identity until a p x p matrix M has eigenvalues of at least tol"""
    p = M.shape[0]
    mineig = np.linalg.eigh(M)[0].min()
    if mineig < tol:
        M += (tol - mineig) * np.eye(p)

    return M


def scale_until_PSD(Sigma, S, tol, num_iter):
    """ 
    Perform a binary search to find the largest ``gamma`` such that the minimum
    eigenvalue of ``2*Sigma - gamma*S`` is at least ``tol``.
    
    Returns
    -------
    gamma * S : np.ndarray
        See description.
    gamma : float
        See description
    """

    # Raise value error if S is not PSD
    try:
        np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        S = shift_until_PSD(S, tol)

    # Binary search to find minimum gamma
    lower_bound = 0
    upper_bound = 1
    for j in range(num_iter):
        gamma = (lower_bound + upper_bound) / 2
        V = 2 * Sigma - gamma * S
        mineig = np.linalg.eigh(V)[0].min()
        if mineig < tol:
            upper_bound = gamma
        else:
            lower_bound = gamma

    # Scale S properly, be a bit conservative
    S = lower_bound * S

    return S, lower_bound


def permute_matrix_by_groups(groups):
    """
    Create indices which permute a (covariance) matrix according to a list of groups.
    """
    # Create sorting indices
    inds_and_groups = [(i, group) for i, group in enumerate(groups)]
    inds_and_groups = sorted(inds_and_groups, key=lambda x: x[1])
    inds = [i for (i, j) in inds_and_groups]

    # Make sure we can unsort
    p = groups.shape[0]
    inv_inds = np.zeros(p)
    for i, j in enumerate(inds):
        inv_inds[j] = i
    inv_inds = inv_inds.astype("int32")

    return inds, inv_inds


def blockdiag_to_blocks(M, groups):
    """
    Given a square array `M`, returns a list of diagonal blocks of `M` as specified by `groups`.

    Parameters
    ----------
    M : np.ndarray
        ``(p, p)``-shaped array
    groups : np.ndarray
        ``(p, )``-shaped array with m unique values. If ``groups[i] == j``,
        this indicates that coordinate ``i`` belongs to group ``j``.

    Returns
    -------
    blocks : list
        A list of square np.ndarrays. blocks[i] corresponds to group identified
        by the ith smallest unique value of ``groups``.
    """
    blocks = []
    for j in np.sort(np.unique(groups)):
        inds = np.where(groups == j)[0]
        full_inds = np.ix_(inds, inds)
        blocks.append(M[full_inds].copy())
    return blocks


### Feature-statistic helpers
def random_permutation_inds(length):
    """ Returns indexes which will randomly permute/unpermute a numpy
    array of length `length`. Also returns indices which will
    undo this permutation.

    Returns
    -------
    inds : np.ndarray
        ``(length,)``-shaped ndarray corresponding to a random permutation
        from 0 to `length`-1.
    rev_inds : np.ndarray
        ``(length,)``-shaped ndarray such that for any ``(length,)``-shaped 
        array called ``x``, ``x[inds][rev_inds]`` equals ``x``.
    """
    # Create inds and rev inds
    inds = np.arange(0, length, 1)
    np.random.shuffle(inds)
    rev_inds = [0 for _ in range(length)]
    for (i, j) in enumerate(inds):
        rev_inds[j] = i

    return inds, rev_inds


### Helper for MX knockoffs when we infer Sigma
def estimate_factor(Sigma, num_factors=20, num_iter=10):
    """
    Approximates ``Sigma = np.diag(D) + np.dot(U, U.T)``.
    See https://arxiv.org/pdf/2006.08790.pdf.

    Parameters
    ----------
    Sigma : np.ndarray
        ``(p, p)``-shaped covariance matrix of X
    num_factors : int
        Dimensionality of ``U``.

    Notes
    -----
    TODO: allow X as an input when Sigma does not
    fit in memory.

    Returns
    -------
    D : np.ndarray
        ``(p,)``-shaped array of diagonal elements.
    U : np.ndarray
        ``(p, num_factors)``-shaped array. 
    """
    p = Sigma.shape[0]
    # Problem is trivial in this case 
    if num_factors >= p:
        return np.zeros((p, p)), sp.linalg.sqrtm(Sigma)

    # Coordinate ascent
    D = np.zeros(p)
    for i in range(num_iter):
        evals, evecs = eigsh(Sigma-np.diag(D), num_factors, which='LM')
        U = np.dot(evecs, np.diag(np.maximum(0, np.sqrt(evals))))
        D = np.diag(Sigma - np.power(U, 2).sum(axis=1))
        #loss = np.power(Sigma - np.diag(D) - np.dot(U, U.T), 2).sum()

    return D, U


def estimate_covariance(X, tol=1e-4, shrinkage="ledoitwolf", **kwargs):
    """ Estimates covariance matrix of X. 

    Parameters
    ----------
    X : np.ndarray
        ``(n, p)``-shaped design matrix
    shrinkage : str
        The type of shrinkage to apply during estimation. One of
        "ledoitwolf", "graphicallasso", or None (no shrinkage). 
    tol : float
        If shrinkage is None but the minimum eigenvalue of the MLE
        is below tol, apply LedoitWolf shrinkage anyway.
    kwargs : dict
        kwargs to pass to the shrinkage estimator.

    Returns
    -------
    Sigma : np.ndarray
        ``(p, p)``-shaped estimated covariance matrix of X
    invSigma : np.ndarray
        ``(p, p)``-shaped estimated precision matrix of X
    """
    Sigma = np.cov(X.T)
    mineig = np.linalg.eigh(Sigma)[0].min()

    # Parse none strng
    if str(shrinkage).lower() == "none" or str(shrinkage).lower() == 'mle':
        shrinkage = None

    # Possibly shrink Sigma
    if mineig < tol or shrinkage is not None:
        # Which shrinkage to use
        if str(shrinkage).lower() == "ledoitwolf" or shrinkage is None:
            ShrinkEst = sklearn.covariance.LedoitWolf(**kwargs)
        elif str(shrinkage).lower() == "graphicallasso":
            kwargs['alpha'] = kwargs.get('alpha', 0.1) # Default regularization
            ShrinkEst = sklearn.covariance.GraphicalLasso(**kwargs)
        else:
            raise ValueError(
                f"Shrinkage arg must be one of None, 'ledoitwolf', 'graphicallasso', not {shrinkage}"
            )

        # Fit shrinkage. Sometimes the Graphical Lasso raises errors
        # so we handle these here.
        try:
            warnings.filterwarnings("ignore")
            ShrinkEst.fit(X)
            warnings.resetwarnings()
        except FloatingPointError:
            warnings.resetwarnings()
            warnings.warn(f"Graphical lasso failed, LedoitWolf matrix")
            ShrinkEst = sklearn.covariance.LedoitWolf()
            ShrinkEst.fit(X)

        # Return
        Sigma = ShrinkEst.covariance_
        invSigma = ShrinkEst.precision_
        return Sigma, invSigma

    # Else return empirical estimate
    return Sigma, chol2inv(Sigma)


### Multiprocessing helper
def _one_arg_function(list_of_inputs, args, func, kwargs):
    """
    Globally-defined helper function for pickling in multiprocessing.
    :param list of inputs: List of inputs to a function
    :param args: Names/args for those inputs
    :param func: A function
    :param kwargs: Other kwargs to pass to the function. 
    """
    new_kwargs = {}
    for i, inp in enumerate(list_of_inputs):
        new_kwargs[args[i]] = inp
    return func(**new_kwargs, **kwargs)


def apply_pool(func, constant_inputs={}, num_processes=1, **kwargs):
    """
    Spawns num_processes processes to apply func to many different arguments.
    This wraps the multiprocessing.pool object plus the functools partial function. 
    
    Parameters
    ----------
    func : function
        An arbitrary function
    constant_inputs : dictionary
        A dictionary of arguments to func which do not change in each
        of the processes spawned, defaults to {}.
    num_processes : int
        The maximum number of processes spawned, defaults to 1.
    kwargs : dict
        Each key should correspond to an argument to func and should
        map to a list of different arguments.

    Returns
    -------
    outputs : list
        List of outputs for each input, in the order of the inputs.

    Examples
    --------
    If we are varying inputs 'a' and 'b', we might have
    ``apply_pool(
        func=my_func, a=[1,3,5], b=[2,4,6]
    )``
    which would return ``[my_func(a=1, b=2), my_func(a=3,b=4), my_func(a=5,b=6)]``.
    """

    # Construct input sequence
    args = sorted(kwargs.keys())
    num_inputs = len(kwargs[args[0]])
    for arg in args:
        if len(kwargs[arg]) != num_inputs:
            raise ValueError(f"Number of inputs differs for {args[0]} and {arg}")
    inputs = [[] for _ in range(num_inputs)]
    for arg in args:
        for j in range(num_inputs):
            inputs[j].append(kwargs[arg][j])

    # Construct partial function
    partial_func = partial(
        _one_arg_function, args=args, func=func, kwargs=constant_inputs,
    )

    # Don't use the pool object if num_processes=1
    num_processes = min(num_processes, len(inputs))
    if num_processes == 1:
        all_outputs = []
        for inp in inputs:
            all_outputs.append(partial_func(inp))
    else:
        with Pool(num_processes) as thepool:
            all_outputs = thepool.map(partial_func, inputs)

    return all_outputs


### Dependency management
def check_kpytorch_available(purpose):
    try:
        import torch
    except ImportError as err:
        raise ValueError(
            f"Pytorch is required for {purpose}, but importing torch raised {err}. See https://pytorch.org/get-started/."
        )
