import warnings
import numpy as np
from . import constants
from . import utilities
from . import mrc
from . import knockoffs
from . import metro
from . import knockoff_stats as kstats


class KnockoffFilter:
    """
    Performs knockoff-based inference, from start to finish.

    This wraps both the ``knockoffs.KnockoffSampler`` and 
    ``knockoff_stats.FeatureStatistic`` classes.

    Parameters 
    ----------
    fstat : str or knockpy.knockoff_stats.FeatureStatistic
        The feature statistic to use in the knockoff filter.
        This may also be a string identifier, including:
        - 'lasso' or 'lcd': cross-validated lasso coefficients differences 
        - 'lsm': signed maximum of the lasso path statistic as 
            in Barber and Candes 2015
        - 'dlasso': Cross-validated debiased lasso coefficients
        - 'ridge': Cross validated ridge coefficients
        - 'ols': Ordinary least squares coefficients
        - 'margcorr': marginal correlations between features and response
        - 'deeppink': The deepPINK statistic as in Lu et al. 2018
        - 'randomforest': A random forest with swap importances
    ksampler : str or knockpy.knockoffs.KnockoffSampler
        The knockoff sampler to use in the knockoff filter.
        This may also be a string identifier, including:
        - 'gaussian': Gaussian Model-X knockoffs
        - 'fx': Fixed-X knockoffs
        - 'metro': Generic metropolized knockoff sampler.
        - 'artk': t-tailed Markov chain
        - 'blockt': Blocks of t-distributed 
        - 'gibbs_grid': Discrete gibbs grid
        An alternative to specifying the ksampler is to simply pass
        in a knockoff matrix during the ``forward`` call.
    fstat_kwargs : dict
        Kwargs to pass to the feature statistic ``fit`` function,
        excluding the required arguments, defaults to {} 
    knockoff_kwargs : dict
        Kwargs for instantiating the knockoff sampler argument if
        the ksampler argument is a string identifier. This can be
        the empt dict for some identifiers such as "gaussian" or "fx",
        but additional keyword arguments are required for complex samplers
        such as the "metro" identifier. Defaults to {}

    Attributes
    ----------
    fstat : knockpy.knockoff_stats.FeatureStatistic
        The feature statistics to use for inference. This inherits
        from ``knockoff_stats.FeatureStatistic``.
    ksampler : knockpy.knockoffs.KnockoffSampler
        The knockoff sampler to use during inference. This eventually
        inherits from ``knockoffs.KnockoffSampler``.
    fstat_kwargs : dict
        Dictionary of kwargs to pass to the ``fit`` call of ``self.fstat``.
    knockoff_kwargs : dict
        If ``ksampler`` is not yet initialized, kwargs to pass to ``ksampler``.
    Z : np.ndarray
        a ``2p``-dimsional array of feature and knockoff importances. The
        first p coordinates correspond to features, the last p correspond
        to knockoffs.
    W : np.ndarray
        an array of feature statistics. This is ``(p,)``-dimensional
        for regular knockoffs and ``(num_groups,)``-dimensional for
        group knockoffs.
    S : np.ndarray
        the ``(p, p)``-shaped knockoff S-matrix used to generate knockoffs.
    X : np.ndarray
        the ``(n, p)``-shaped design matrix
    Xk : np.ndarray
        the ``(n, p)``-shaped matrix of knockoffs
    groups : np.ndarray
        For group knockoffs, a p-length array of integers from 1 to 
        num_groups such that ``groups[j] == i`` indicates that variable `j`
        is a member of group `i`. Defaults to None (regular knockoffs).
    rejections : np.ndarray
        a ``(p,)``-shaped boolean array where rejections[j] == 1 iff the
        the knockoff filter rejects the jth feature.
    G : np.ndarray
        the ``(2p, 2p)``-shaped feature-knockoff covariance matrix
    threshold : float
        the knockoff data-dependent threshold used to select variables

    Examples
    --------
    Here we fit the KnockoffFilter on fake data from a Gaussian
    linear model: ::

        # Fake data-generating process for Gaussian linear model
        import knockpy as kpy
        dgprocess = kpy.dgp.DGP()
        dgprocess.sample_data(n=500, p=500, sparsity=0.1)

        # LCD statistic with Gaussian MX knockoffs
        # This uses LedoitWolf covariance estimation by default.
        from knockpy.knockoff_filter import KnockoffFilter 
        kfilter = KnockoffFilter( 
            fstat='lcd', 
            ksampler='gaussian', 
            knockoff_kwargs={"method":"mvr"}, 
        )
        rejections = kfilter.forward(X=dgprocess.X, y=dgprocess.y)
    """

    def __init__(
        self, fstat="lasso", ksampler="gaussian", fstat_kwargs={}, knockoff_kwargs={},
    ):
        """
        Initialize the class.
        """

        ### Parse feature statistic
        self.fstat_kwargs = fstat_kwargs.copy()
        if isinstance(fstat, str):
            fstat = fstat.lower()
        # Save feature statistic if class-based
        if isinstance(fstat, kstats.FeatureStatistic):
            pass
        # Else parse flags
        elif fstat == "lasso" or fstat == "lcd":
            fstat = kstats.LassoStatistic()
        elif fstat == "lsm":
            fstat = kstats.LassoStatistic()
            self.fstat_kwargs["zstat"] = "lars_path"
            self.fstat_kwargs["antisym"] = "sm"
        elif fstat == "dlasso":
            fstat = kstats.LassoStatistic()
            self.fstat_kwargs["debias"] = True
        elif fstat == "ridge":
            fstat = kstats.RidgeStatistic()
        elif fstat == "ols":
            fstat = kstats.OLSStatistic()
        elif fstat == "margcorr":
            fstat = kstats.MargCorrStatistic()
        elif fstat == "randomforest":
            fstat = kstats.RandomForestStatistic()
        elif fstat == "deeppink":
            fstat = kstats.DeepPinkStatistic()
        else:
            raise ValueError(f"Unrecognized fstat {fstat}")
        self.fstat = fstat

        ### Preprocessing for knockoffs
        self.knockoff_kwargs = knockoff_kwargs.copy()
        if isinstance(ksampler, str):
            self.knockoff_type = ksampler.lower()
            self.ksampler = None
        if isinstance(ksampler, knockoffs.KnockoffSampler):
            self.knockoff_type = None
            self.ksampler = ksampler
        if isinstance(ksampler, knockoffs.FXSampler):
            self._mx = False
        elif self.knockoff_type == "fx":
            self._mx = False
        else:
            self._mx = True

        # Initialize
        self.S = None

    def sample_knockoffs(self):
        """
        Samples knockoffs during ``forward``.
        """

        # If we have already computed S, signal this
        # because this is expensive
        if self.S is not None:
            if "S" not in self.knockoff_kwargs:
                self.knockoff_kwargs["S"] = self.S

        # Possibly initialize ksampler
        if self.ksampler is None:
            # Args common to all ksamplers except fx
            args = {"X": self.X, "Sigma": self.Sigma}
            if self.knockoff_type == "gaussian":
                self.ksampler = knockoffs.GaussianSampler(
                    groups=self.groups, mu=self.mu, **args, **self.knockoff_kwargs
                )
            elif self.knockoff_type == "fx":
                self.ksampler = knockoffs.FXSampler(
                    X=self.X, groups=self.groups, **self.knockoff_kwargs,
                )
            elif self.knockoff_type == "artk":
                self.ksampler = metro.ARTKSampler(**args, **self.knockoff_kwargs,)
            elif self.knockoff_type == "blockt":
                self.ksampler = metro.BlockTSampler(**args, **self.knockoff_kwargs,)
            elif self.knockoff_type == "metro":
                self.ksampler = metro.MetropolizedKnockoffSampler(
                    **args, mu=self.mu, **self.knockoff_kwargs
                )
            elif self.knockoff_type == "gibbs_grid":
                self.ksampler = metro.GibbsGridSampler(
                    **args, mu=self.mu, **self.knockoff_kwargs
                )
            else:
                raise ValueError(f"Unrecognized ksampler string {self.knockoff_type}")
        Xk = self.ksampler.sample_knockoffs()
        self.S = self.ksampler.fetch_S()

        # Possibly use recycling
        if self.recycle_up_to is not None:
            # Split
            rec_Xk = self.X[: self.recycle_up_to]
            new_Xk = Xk[self.recycle_up_to :]
            # Combine
            Xk = np.concatenate((rec_Xk, new_Xk), axis=0)
        self.Xk = Xk

        # Construct the feature-knockoff covariance matrix, or estimate
        # it if construction is not possible
        if self.S is not None and self.Sigma is not None:
            self.G = np.concatenate(
                [
                    np.concatenate([self.Sigma, self.Sigma - self.S]),
                    np.concatenate([self.Sigma - self.S, self.Sigma]),
                ],
                axis=1,
            )
            # Handle errors where Ginv is exactly low rank
            try:
                self.Ginv = utilities.chol2inv(self.G)
            except np.linalg.LinAlgError:
                warnings.warn("The feature-knockoff covariance matrix is low rank.")
                self.Ginv = None
        else:
            self.G, self.Ginv = utilities.estimate_covariance(
                np.concatenate([self.X, self.Xk], axis=1)
            )

        return self.Xk

    def make_selections(self, W, fdr):
        """" Calculate data dependent threshhold and selections """
        self.threshold = kstats.data_dependent_threshhold(W=W, fdr=fdr)
        selected_flags = (W >= self.threshold).astype("float32")
        return selected_flags

    def forward(
        self,
        X,
        y,
        Xk=None,
        mu=None,
        Sigma=None,
        groups=None,
        fdr=0.10,
        fstat_kwargs={},
        knockoff_kwargs={},
        shrinkage="ledoitwolf",
        num_factors=None,
        recycle_up_to=None,
    ):
        """
        Runs the knockoff filter; returns whether each feature was rejected.
        
        Parameters
        ----------
        X : np.ndarray
            ``(n, p)``-shaped design matrix
        y : np.ndarray
            ``(n,)``-shaped response vector
        Xk : np.ndarray
            ``(n, p)``-shaped knockoff matrix. If ``None``, this will construct
            knockoffs using ``self.ksampler``.
        mu : np.ndarray
            ``(p, )``-shaped mean of the features. If ``None``, this defaults to
            the empirical mean of the features.
        Sigma : np.ndarray
            ``(p, p)``-shaped covariance matrix of the features. If ``None``, this
            is estimated using the ``shrinkage`` option. This is ignored for
            fixed-X knockoffs.
        groups : np.ndarray
            For group knockoffs, a p-length array of integers from 1 to 
            num_groups such that ``groups[j] == i`` indicates that variable `j`
            is a member of group `i`. Defaults to ``None`` (regular knockoffs). 
        fdr : float
            The desired level of false discovery rate control.
        fstat_kwargs : dict
            Extra kwargs to pass to the feature statistic ``fit`` function,
            excluding the required arguments.
        knockoff_kwargs : dict
            Extra kwargs for instantiating the knockoff sampler argument if
            the ksampler argument is a string identifier. This can be
            the empty dict for some identifiers such as "gaussian" or "fx",
            but additional keyword arguments are required for complex samplers
            such as the "metro" identifier. Defaults to {}
        shrinkage : str
            Shrinkage method if estimating the covariance matrix. Defaults to 
            "LedoitWolf." Other options are "MLE" and "glasso" (graphical lasso).
        num_factors : int
            If num_factors is not ``None`` and Sigma is estimated,
            assumes that the ground-truth Sigma is a factor model 
            with rank ``num_factors`` to speed up computation.
        recycle_up_to : int or float
            Three options:
                - if ``None``, does nothing.
                - if an integer > 1, uses the first "recycle_up_to"
                rows of X as the the first ``recycle_up_to`` rows of knockoffs
                - if a float between 0 and 1 (inclusive), interpreted
                as the proportion of rows to recycle. 
            For more on recycling, see https://arxiv.org/abs/1602.03574
        """

        # Preliminaries - infer covariance matrix for MX
        if Sigma is None and self._mx:
            Sigma, _ = utilities.estimate_covariance(X, 1e-2, shrinkage)
            # Possible factor model approximation
            if num_factors is not None and Sigma is not None:
                self.D, self.U = utilities.estimate_factor(
                    Sigma, num_factors=num_factors
                )
                Sigma = np.diag(self.D) + np.dot(self.U, self.U.T)
                self.knockoff_kwargs['how_approx'] = 'factor'
                self.knockoff_kwargs['D'] = self.D
                self.knockoff_kwargs['U'] = self.U
            else:
                self.D = None
                self.U = None
        if not self._mx:
            Sigma = None


        # Save objects
        self.X = X
        self.Xk = Xk
        self.y = y
        self.mu = mu
        self.Sigma = Sigma
        self.groups = groups
        for key in fstat_kwargs:
            self.fstat_kwargs[key] = fstat_kwargs[key]
        for key in knockoff_kwargs:
            self.knockoff_kwargs[key] = knockoff_kwargs[key]

        # Save n, p, groups
        n = X.shape[0]
        p = X.shape[1]
        if groups is None:
            groups = np.arange(1, p + 1, 1)

        # Parse recycle_up_to
        if recycle_up_to is None:
            pass
        elif recycle_up_to < 1:
            recycle_up_to = int(recycle_up_to * n)
        else:
            recycle_up_to = int(recycle_up_to)
        self.recycle_up_to = recycle_up_to

        # Sample knockoffs
        if self.Xk is None:
            self.Xk = self.sample_knockoffs()

        # As an edge case, pass Ginv to debiased lasso
        if "debias" in self.fstat_kwargs:
            if self.fstat_kwargs["debias"]:
                self.fstat_kwargs["Ginv"] = self.Ginv

        # Feature statistics
        self.fstat.fit(
            X=self.X, Xk=self.Xk, y=self.y, groups=self.groups, **self.fstat_kwargs
        )
        # Inherit some attributes
        self.Z = self.fstat.Z
        self.W = self.fstat.W
        self.score = self.fstat.score
        self.score_type = self.fstat.score_type
        self.rejections = self.make_selections(self.W, fdr)

        # Return
        return self.rejections
