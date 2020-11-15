import numpy as np
from . import utilities
from . import mrc
from . import knockoffs
from . import metro
from . import knockoff_stats as kstats 

class KnockoffFilter:
    """
    Knockoff Filter class.
    :param fstat: The feature statistic 
    """
    def __init__(
            self,
            fstat='lasso',
            ksampler='gaussian',
            fstat_kwargs={},
            knockoff_kwargs={},
        ):
        """
        :param fstat: The feature statistic to use in the knockoff
        filter. This may be a string identifying a specific type of
        feature statistic like the 'lasso' or an initialized class inheriting from
        the base FeatureStatistic class. 
        Options for string identifiers include:
            - 'lasso' or 'lcd': cross-validated lasso coefficients differences.
            - 'lsm': signed maximum of the lasso path statistic as
            in Barber and Candes 2015.
            - 'dlasso': Cross-validated debiased lasso coefficients
            - 'ridge': Cross validated ridge coefficients
            - 'ols': Ordinary least squares coefficients
            - 'margcorr': marginal correlations between features and response
            - 'deeppink': The deepPINK statistic as in Lu et al. 2018
            - 'randomforest': A random forest statistic with swap importances
        :param ksampler: The method for sampling knockoffs. This may be
        a class inheriting from KnockoffSampler or a string identifying 
        a specific knockoff sampling strategy. String identifiers include:
            - 'gaussian': Gaussian MX knockoffs
            - 'fx': Fixed-X knockoffs
            - 'artk': t-tailed Markov chain
            - 'blockt': Blocks of t-distributed 
            - 'ising': Discrete gibbs grid
        :param fstat_kwargs: Kwargs to pass to the feature statistic
        fit function. 
        :param knockoff_kwargs: Kwargs for instantiating the knockoff sampler
        argument.
        """

        ### Parse feature statistic
        self.fstat_kwargs = fstat_kwargs.copy()
        if isinstance(fstat, str):
            fstat = fstat.lower()
        # Save feature statistic if class-based
        if isinstance(fstat, kstats.FeatureStatistic):
            pass
        # Else parse flags
        elif fstat == "lasso" or fstat == 'lcd':
            fstat = kstats.LassoStatistic()
        elif fstat == 'lsm':
            fstat = kstats.LassoStatistic()
            self.fstat_kwargs['zstat'] = 'lars_path'
            self.fstat_kwargs['pair_agg'] = 'sm'
        elif fstat == 'dlasso':
            fstat = kstats.LassoStatistic()
            self.fstat_kwargs['debias'] = True
        elif fstat == 'ridge':
            fstat = kstats.RidgeStatistic()
        elif fstat == "ols":
            fstat = kstats.OLSStatistic()
        elif fstat == "margcorr":
            fstat = kstats.MargCorrStatistic()
        elif fstat == 'randomforest':
            fstat = kstats.RandomForestStatistic()
        elif fstat == 'deeppink':
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
            self.mx = False
        elif self.knockoff_type == 'fx':
            self.mx = False
        else:
            self.mx = True

        # Initialize
        self.S = None



    def sample_knockoffs(self):

        # If we have already computed S, signal this
        # because this is expensive
        if self.S is not None:
            if 'S' not in self.knockoff_kwargs:
                self.knockoff_kwargs['S'] = self.S

        # Possibly initialize ksampler
        if self.ksampler is None:
            # Args common to all ksamplers
            args = {
                'X':self.X,
                'Sigma':self.Sigma
            }
            if self.knockoff_type == 'gaussian':
                self.ksampler = knockoffs.GaussianSampler(
                    groups=self.groups,
                    mu=self.mu,
                    **args,
                    **self.knockoff_kwargs
                )
            elif self.knockoff_type == 'fx':
                self.ksampler = knockoffs.FXSampler(
                    X=self.X,
                    groups=self.groups,
                    **self.knockoff_kwargs,
                )
            elif self.knockoff_type == 'artk':
                self.ksampler = metro.ARTKSampler(
                    **args,
                    **self.knockoff_kwargs,
                )
            elif self.knockoff_type == 'blockt':
                self.ksampler = metro.BlockTSampler(
                    **args, 
                    **self.knockoff_kwargs,
                )
            elif self.knockoff_type == 'ising':
                if 'gibbs_graph' not in self.knockoff_kwargs:
                    raise ValueError(
                        f"For ising knockoffs, must provide gibbs graph as knockoff_kwarg"
                    )
                self.ksampler = metro.IsingKnockoffSampler(
                    **args,
                    mu=self.mu,
                    **self.knockoff_kwargs
                )
            else:
                raise ValueError(
                    f"Unrecognized ksampler string {self.knockoff_type}"
                )
        Xk = self.ksampler.sample_knockoffs()
        self.S = self.ksampler.fetch_S()

        # Possibly use recycling
        if self.recycle_up_to is not None:
            # Split
            rec_Xk = self.X[:self.recycle_up_to]
            new_Xk = Xk[self.recycle_up_to:]
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
            except np.errors.LinAlgError:
                warnings.warn("The feature-knockoff covariance matrix is low rank.")
                self.Ginv = None
        else:
            self.G, self.Ginv = utilities.estimate_covariance(
                np.concatenate([self.X, self.Xk], axis=1)
            )

        return self.Xk

    def make_selections(self, W, fdr):
        """" Calculate data dependent threshhold and selections """
        T = kstats.data_dependent_threshhold(W=W, fdr=fdr)
        selected_flags = (W >= T).astype("float32")
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
        shrinkage='ledoitwolf',
        recycle_up_to=None,
    ):
        """
        :param X: n x p design matrix
        :param y: p-length response array
        :param Sigma: p x p covariance matrix of X. Defaults to None
        for FX knockoffs or 
        :param groups: Grouping of features, p-length
        array of integers from 1 to m (with m <= p).
        :param knockoffs: n x p array of knockoffs.
        If None, will construct second-order group MX knockoffs.
        Defaults to group gaussian knockoff constructor.
        :param feature_stat: Function used to
        calculate W-statistics in knockoffs. 
        Defaults to group lasso coefficient difference.
        :param fdr: Desired fdr.
        :param feature_stat: A classname with a fit method.
        The fit method must takes X, knockoffs, y, and groups,
        and returns a set of p anti-symmetric knockoff 
        statistics. Can also be one of "lasso", "ols", or "margcorr." 
        :param feature_stat_kwargs: Kwargs to pass to 
        the feature statistic.
        :param knockoff_kwargs: Kwargs to pass to the 
        knockoffs constructor.
        :param shrinkage: Shrinkage method if estimating the covariance
        matrix. Defaults to "LedoitWolf."
        :param recycle_up_to: Three options:
            - if None, does nothing.
            - if an integer > 1, uses the first "recycle_up_to"
            rows of X as the the first "recycle_up_to" rows of knockoffs.
            - if a float between 0 and 1 (inclusive), interpreted
            as the proportion of knockoffs to recycle. 
        For more on recycling, see https://arxiv.org/abs/1602.03574
        """

        # Preliminaries - infer covariance matrix for MX
        if Sigma is None and self.mx:
            Sigma, _ = utilities.estimate_covariance(X, 1e-2, shrinkage)

        # Save objects
        self.X = X
        self.Xk = Xk
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
        if 'debias' in self.fstat_kwargs:
            if self.fstat_kwargs['debias']:
                self.fstat_kwargs['Ginv'] = self.Ginv

        # Feature statistics
        self.fstat.fit(
            X=self.X, Xk=self.Xk, y=y, groups=groups, **self.fstat_kwargs
        )
        # Inherit some attributes
        self.Z = self.fstat.Z
        self.W = self.fstat.W
        self.score = self.fstat.score
        self.score_type = self.fstat.score_type
        self.selected_flags = self.make_selections(self.W, fdr)

        # Return
        return self.selected_flags