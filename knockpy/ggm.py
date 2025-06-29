"""
Knockoffs for detecting edges in gaussian graphical models.
See https://arxiv.org/pdf/1908.11611.pdf.
"""

import copy

import numpy as np

from .knockoff_filter import KnockoffFilter as KF


def discovered_edges(W, T, logic="and"):
    """
    Parameters
    ----------
    W : np.array
            ``(p,p)``-shaped array of knockoff statistics
            with zeros along the diagonal. The rows of W must obey
            the flip-sign property.
    T : np.array
            ``(p,)``-shaped array of knockoff thresholds
    logic : string
            One of 'and' or 'or'. This is a hyperparameter used to
            determine the rejection set. Defaults to "and".

    Returns
    -------
    edges : np.array
            ``(p,p)``-shaped symmetric boolean array where
            edges[i,j] is true if and only if edge (i,j) has been
            discovered.
    """
    edges = W >= T.reshape(-1, 1)
    if logic == "and":
        edges = edges & edges.T
    elif logic == "or":
        edges = edges | edges.T
    else:
        raise ValueError(f"logic={logic} must be one of 'and', 'or'")
    return edges


def compute_ggm_threshold(W, fdr=0.1, logic="and", a=1, offset=1):
    """
        Parameters
        ----------
        W : np.array
                ``(p,p)``-shaped array of knockoff statistics
                with zeros along the diagonal. The rows of W must obey
                the flip-sign property.
        fdr : float
                Desired level of FDR control.
        logic : string
                One of 'and' or 'or'. This is a hyperparameter used to
                determine the rejection set. Defaults to "and".
        a : float
                One of 0.01 or 1.
                Hyperparameter used to determine the rejection threshold.
                See Li and Maathuis (2019) for discussion.
    offset : int
        If offset = 0, control the modified FDR.
        If offset = 1 (default), controls the FDR exactly.

        Returns
        -------
        T : np.array
                ``(p,)``-shaped array of knockoff thresholds

        Notes
        -----
        See Algorithm 2, https://arxiv.org/pdf/1908.11611.pdf

    """
    p = W.shape[0]
    eps = W[W > 0].min() / 2  # smallest threshold
    if np.any(np.diag(W) != 0):
        raise ValueError("W matrix must have a diagonal of all zeros")
    if a == 1.0:
        ca = 1.93
    elif a == 0.01:
        ca = 102
    else:
        raise ValueError(f"a={a} must be one of [0.01,1.0]")

    # Compute start of loop
    logic = str(logic).lower()
    if logic == "and":
        mmax = fdr * (p - 1) / ca - a * offset
        qi = 2 * fdr / (ca * p)
    elif logic == "or":
        mmax = fdr * (p - 1) / (2 * ca) - a * offset
        qi = fdr / (ca * p)
    else:
        raise ValueError(f"logic={logic} must be one of 'and', 'or'")
    mmax = int(np.floor(mmax))
    if mmax < 0:
        return np.zeros(p) + np.inf

    # Preprocess signs
    inds = np.argsort(-1 * np.abs(W), axis=1, stable="stable")
    sortW = np.take_along_axis(W, inds, axis=1)
    np.cumsum(sortW > 0, axis=1)
    cumneg = np.cumsum(sortW < 0, axis=1)

    # Loop through thresholds. `ms` indexes the number
    # of negatives allowed.
    ms = np.flip(np.arange(0, mmax + 1, 1))
    for m in ms:
        # This gives the last index less than or equal to m
        Tinds = np.argmin(cumneg <= m, axis=1) - 1
        # Construct thresholds based on Tinds
        T = np.abs(sortW[np.arange(p), Tinds])
        # if all indices are <= m
        all_leq_m = np.all(cumneg <= m, axis=1)
        T[Tinds == -1] = np.inf  # if no index is less than or equal to m
        T[(Tinds == -1) & all_leq_m] = eps
        # Create set of discovered edges
        edges = discovered_edges(W=W, T=T, logic=logic)
        ndisc = edges.sum()  # is this double-counting? I don't think so, but am not sure. If so divide by 2.
        # print(m, (a + m) / max(ndisc, 1.0), qi)
        if (a + m) / max(ndisc, 1.0) < qi:
            return T

    # If no feasible solution, return nothing
    return np.zeros(p) + np.inf


class KnockoffGGM:
    """
        Tests for edges in a Gaussian Graphical Model.
        See Li and Maathuis (2019) for details.

        Parameters
        ----------
        fstat : str
        The feature statistic to use in the knockoff filter: this
        must be a string and must be a valid fixed-X knockoff
        feature statistic. Identifiers include:
        - 'lasso' or 'lcd': lasso coefficients differences
        - 'lsm': signed maximum of the lasso path statistic as
            in Barber and Candes 2015
        - 'ols': Ordinary least squares coefficients
        - 'margcorr': marginal correlations between features and response
    fstat_kwargs : dict
        Kwargs to pass to the feature statistic ``fit`` function,
        excluding the required arguments, defaults to {}
    knockoff_kwargs : dict
        Kwargs for instantiating the knockoff sampler argument if
        the ksampler argument is a string identifier. Defaults to {}

        Attributes
        ----------
        W : np.array
                ``(p,p)``-shaped array of knockoff statistics
                with zeros along the diagonal. The rows of W obey
                the flip-sign property, i.e., W[0] obeys the flip-sign
                property.
        kfs : list
                A list of KnockoffFilter classes corresponding to the regression
                run on each covariate.

    Notes
    -----
    There is no known way to use model-X knockoffs for this application.


    Examples
    --------
    Here we fit KnockoffGGM under the global null when the true Gaussian
    graphical model has no edges:

        # Fake data-generating process for Gaussian graphical model
        import numpy as np
        X = np.random.randn(300, 30)

        # LCD statistic with FX knockoffs
        from knockpy.ggm import KnockoffGGM
        gkf = KnockoffGGM(
            fstat='lcd',
            knockoff_kwargs={"method":"mvr"},
        )
        edges = gkf.forward(X=X, verbose=True)
    """

    def __init__(self, fstat="lcd", fstat_kwargs=None, knockoff_kwargs=None):
        self.fstat = fstat
        self.fstat_kwargs = fstat_kwargs if fstat_kwargs is not None else {}
        self.knockoff_kwargs = knockoff_kwargs if knockoff_kwargs is not None else {}

    def forward(self, X, logic="and", fdr=0.1, ggm_kwargs=None, verbose=True):
        """
                Runs the GGM filter by applying fixed-X knockoffs
                to each column of X using the other columns as covariates.

                Parameters
                ----------
        X : np.ndarray
            ``(n, p)``-shaped design matrix.
        fdr : float
                Nominal level at which to control the FDR.
                logic : string
                        One of 'and' or 'or'. This is a hyperparameter used to
                        determine the rejection set. Defaults to "and".
        ggm_kwargs : dict
                Dictionary of hyperparameters to pass to the
                ``ggm.compute_ggm_threshold`` function. Defaults to {}.
        verbose : bool
                If true, log progress over time.

                Returns
                -------
                edges : np.array
                        ``(p,p)``-shaped symmetric boolean array where
                        edges[i,j] is true if and only if edge (i,j) has been
                        discovered.

        Notes
        -----
        This requires fitting knockoffs p times, so it is quite expensive.
        """
        self.n, self.p = X.shape
        self.X = X
        self.logic = logic
        self.fdr = fdr
        self.ggm_kwargs = ggm_kwargs if ggm_kwargs is not None else {}
        self.kfs = []
        self.Ws = np.zeros((self.p, self.p))
        # Loop through columns to fit knockoffs
        for j in range(self.p):
            kf = KF(
                ksampler="fx",
                fstat=self.fstat,
                knockoff_kwargs=copy.copy(self.knockoff_kwargs),
                fstat_kwargs=copy.copy(self.fstat_kwargs),
            )
            negj = [i for i in range(self.p) if i != j]
            Xnegj = X[:, negj]
            kf.forward(
                X=Xnegj,
                y=X[:, j],
            )
            self.kfs.append(kf)
            self.Ws[j][negj] = kf.W
            if verbose:
                print(f"Finished feature {j} of {self.p}.")
        # Find threshold
        self.T = compute_ggm_threshold(W=self.Ws, logic=self.logic, **self.ggm_kwargs)
        # Compute rejections
        self.edges = discovered_edges(W=self.Ws, T=self.T, logic=self.logic)
        return self.edges
