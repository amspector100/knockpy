import numpy as np
import scipy.special

from .. import knockoff_stats as kstats
from .. import utilities
from ._mlr_oracle import _sample_mlr_oracle_gaussian, _sample_mlr_oracle_logistic
from ._mlr_spikeslab import _sample_mlr_spikeslab
from ._mlr_spikeslab_fx import _sample_mlr_spikeslab_fx
from ._mlr_spikeslab_group import _sample_mlr_spikeslab_group


def check_no_groups(groups, p, error=None):
    if groups is not None:
        if np.any(groups != np.arange(1, p + 1)):
            if error is not None:
                raise ValueError(error)
            return False
    return True

def _calc_group_blocks(groups, group_sizes):
    max_gsize = np.max(group_sizes)
    ngroups = group_sizes.shape[0]
    group_blocks = -1 * np.ones((ngroups, max_gsize), dtype=int)
    for gj in range(ngroups):
        gsize = group_sizes[gj]
        group_blocks[gj, 0:gsize] = np.where(groups == (gj + 1))[0]
    return group_blocks.astype(int)

def _mlr_to_adj_mlr(mlr_sign, prob_mlr_pos, prob_mlr_pos_nonnull, fdr):
    mlr_abs = np.log(prob_mlr_pos / (1 - prob_mlr_pos))
    # threshold
    thresh = 1 / (1+fdr)
    thresh = np.log(thresh / (1 - thresh))
    # Adjusted statistics
    nu_ratio = prob_mlr_pos_nonnull / (thresh - prob_mlr_pos)
    amlr_abs = mlr_abs.copy()
    amlr_abs[mlr_abs <= thresh] = thresh * scipy.special.expit(nu_ratio[mlr_abs <= thresh])
    # Final statistic
    W = amlr_abs * mlr_sign
    # Clip numerical errors and return
    W[np.abs(W) < 1e-15] = 0
    return W

class MLR_Spikeslab(kstats.FeatureStatistic):
    """
    Masked likelihood ratio statistics using a spike-and-slab
    prior for a linear model or probit model, automatically
    inferred based on y.

    Parameters
    ----------
    X : np.ndarray
            the ``(n, p)``-shaped design matrix
    Xk : np.ndarray
            the ``(n, p)``-shaped matrix of knockoffs
    y : np.ndarray
            ``(n,)``-shaped response vector
    n_iter : int
            Number of samples per MCMC chain used to compute
            MLR statistics. Default: 2000.
    chains : int
            Number of MCMC chains to run. Default: 5.
    burn_prop : float
            The burn-in for each chain will be equal to
            ``n_iter * burn_prop``.
    adjusted_mlr : bool
            If True, computes adjusted MLR (AMLR) statistics. Default: False.
            See the MLR knockoffs paper for details.
    p0 : float
            Prior probability that any coefficient equals zero.
    update_p0 : bool
            If True, updates ``p0`` using a Beta hyperprior on ``p0``.
            Else, the value of ``p0`` is fixed. Default: True.
    p0_a0 : float
            If ``update_p0`` is True, ``p0`` has a
            TruncBeta(``p0_a0``, ``p0_b0``, ``min_p0``) hyperprior.
            Default: 1.0.
    p0_b0 : float
            If ``update_p0`` is True, ``p0`` has a
            TruncBeta(``p0_a0``, ``p0_b0``, ``min_p0``) hyperprior.
            Default: 1.0.
    min_p0 : float
            Minimum value for ``p0`` as specified by the prior.
            Default: 0.8.
    sigma2 : float
            Variance of y given X. Default: 1.0.
    update_sigma2 : bool
            If True, imposes an InverseGamma hyperprior on ``sigma2``.
            Else, the value of ``sigma2`` is fixed. Default: True.
    sigma2_a0 : float
            If ``update_sigma2`` is True, ``sigma2`` has an
            InvGamma(``sigma2_a0``, ``sigma2_b0``) hyperprior.
            Default: 2.0.
    sigma2_b0 : float
            If ``update_sigma2`` is True, ``sigma2`` has an
            InvGamma(``sigma2_a0``, ``sigma2_b0``) hyperprior.
            Default: 1.0.
    tau2 : float
            Prior variance on nonzero coefficients. Default: 1.0.
    update_tau2 : bool
            If True, imposes an InverseGamma hyperprior on ``tau2``.
            Else, the value of ``tau2`` is fixed. Default: True.
    tau2_a0 : float
            If ``update_tau2`` is True, ``tau2`` has an
            InvGamma(``tau2_a0``, ``tau2_b0``) hyperprior.
            Default: 2.0.
    tau2_b0 : float
            If ``update_tau2`` is True, ``tau2`` has an
            InvGamma(``tau2_a0``, ``tau2_b0``) hyperprior.
            Default: 1.0.

    Returns
    -------
    W : np.ndarray
        a ``p``-dimensional array of feature statistics.

    Notes
    -----
    This is a valid FX feature statistic (obeys the sufficiency property).
    """
    def __init__(self, **kwargs):
        # dummy attributes
        self.Z = None
        self.score = None
        self.score_type = None
        self.kwargs = kwargs
        self.kwargs["n_iter"] = self.kwargs.get("n_iter", 2000)
        self.kwargs["burn_prop"] = self.kwargs.get("burn_prop", 0.1)
        self.kwargs["chains"] = self.kwargs.get("chains", 5)
        self.adjusted_mlr = self.kwargs.pop("adjusted_mlr", False)
        self.fdr = self.kwargs.pop("fdr", 0.05)

    def fit(
        self, X, Xk, y, groups, **kwargs
    ):
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.groups = groups
        self.ungrouped = check_no_groups(self.groups, self.p)
        if not self.ungrouped:
            self.group_sizes = utilities.calc_group_sizes(self.groups).astype(int)
            self.group_blocks = _calc_group_blocks(self.groups, self.group_sizes)
            self.ngroup = len(self.group_sizes)
        else:
            self.ngroup = self.p

        self.features = np.concatenate([X, Xk], axis=1)
        for key in kwargs:
            self.kwargs[key] = kwargs[key]

        # kwargs that cannot be passed to the underlying cython
        self.n_iter = self.kwargs.pop("n_iter", 2000)
        self.chains = self.kwargs.pop("chains", 5)
        self.N = int(self.n_iter * self.chains)
        self.burn = int(self.kwargs.pop("burn_prop", 0.1) * self.n_iter)

        # Defaults for model-X
        self.kwargs['tau2_a0'] = self.kwargs.get('tau2_a0', 2.0)
        self.kwargs['tau2_b0'] = self.kwargs.get('tau2_a0', 1.0)
        self.kwargs['sigma2_a0'] = self.kwargs.get('tau2_a0', 2.0)
        self.kwargs['sigma2_b0'] = self.kwargs.get('tau2_a0', 1.0)
        self.kwargs['min_p0'] = self.kwargs.get('min_p0', 0.8)

        # Check whether this is binary or linear regression
        support = np.unique(y)
        if len(support) == 2:
            probit = int(1)
            z = (y == support[0]).astype(int)
        else:
            probit = int(0)
            z = np.zeros(self.n).astype(int)

        # Posterior sampling
        all_out = []
        for chain in range(self.chains):
            if self.ungrouped:
                out = _sample_mlr_spikeslab(
                    N=self.n_iter + self.burn,
                    features=self.features,
                    y=y.astype(np.float64),
                    z=z,
                    probit=probit,
                    **self.kwargs
                )
            else:
                self.groups -= 1
                out = _sample_mlr_spikeslab_group(
                    N=self.n_iter + self.burn,
                    features=self.features,
                    groups=self.groups,
                    blocks=self.group_blocks,
                    gsizes=self.group_sizes,
                    max_gsize=np.max(self.group_sizes).astype(int),
                    y=y.astype(np.float64),
                    z=z,
                    probit=probit,
                    # **self.kwargs
                )
                self.groups += 1
            all_out.append(out)
        self.betas = np.concatenate([x['betas'][self.burn:] for x in all_out])
        #self.beta_logodds = np.concatenate([x['beta_logodds'][self.burn:] for x in all_out])
        self.etas = np.concatenate([x['etas'][self.burn:] for x in all_out])
        self.p0s = np.concatenate([x['p0s'][self.burn:] for x in all_out])
        self.psis = np.concatenate([x['psis'][self.burn:] for x in all_out])
        self.tau2s = np.concatenate([x['tau2s'][self.burn:] for x in all_out])
        self.sigma2s = np.concatenate([x['sigma2s'][self.burn:] for x in all_out])
        if probit == 1:
            self.y_latents = np.concatenate([x['y_latent'][self.burn:] for x in all_out])
        return self.compute_W()

    def compute_W(self):
        # Compute default MLR statistics
        if not self.adjusted_mlr:
            # Compute P(choose feature)
            # etas = log(P(choose feature) / P(choose knockoff))
            etas_cat = np.concatenate(
                [
                    self.etas.reshape(self.N, self.ngroup, 1),
                    np.zeros((self.N, self.ngroup, 1))
                ],
                axis=2
            )
            # this equals log(P(choose feature))
            log_prob = scipy.special.log_softmax(etas_cat, axis=2)[:, :, 0]
            self.log_prob = scipy.special.logsumexp(log_prob, b=1/self.N, axis=0)
            self.W = np.exp(self.log_prob) - 0.5
            # clip numerical errors
            self.W[np.abs(self.W) < 1e-15] = 0
            return self.W
        # Else, compute adjusted MLR statistics.
        if self.adjusted_mlr:
            tol = 1e-10
            # Compute sign guess based on psis
            guess = np.mean(self.psis, axis=0)
            guess[guess > 0.5] = 1
            guess[guess < 0.5] = 0
            guess[guess == 0.5] = np.random.binomial(1, 0.5, size=np.sum(guess == 0.5))
            # Compute P_j(MLR_j > 0 | D)
            mlr_pos = np.clip(np.mean(self.psis == guess, axis=0), tol, 1-tol)
            # Compute P(MLR_j > 0, j is non-null | masked data)
            mlr_pos_nonnull = np.clip(
                np.mean((self.psis == guess) * (self.betas != 0), axis=0), tol, 1-tol
            )
            # MLR signs
            mlr_sign = 2 * (guess == 0) - 1
            # AMLR stats
            self.W = _mlr_to_adj_mlr(
                mlr_sign=mlr_sign, 
                prob_mlr_pos=mlr_pos, 
                prob_mlr_pos_nonnull=mlr_pos_nonnull, 
                fdr=self.fdr,
            )
            return self.W


        
class MLR_Spikeslab_Splines(MLR_Spikeslab):
    """
    Masked likelihood ratio statistics using a spike-and-slab
    prior for a linear or probit model based on regression splines.

    All parameters are the same as MLR_Spikeslab except the 
    following additional parameters.

    Parameters
    ----------
    n_knots : int
        The number of knots to used for the regression splines.
        Defaults to 1.
    degree : int
        The number of degrees to use for the regresison splines.
        Defaults to 3.
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def fit(self, X, Xk, groups, y, **kwargs):
        self.X = X
        self.Xk = Xk
        self.n, self.p = self.X.shape
        self.y = y
        self.groups = groups
        if self.groups is None:
            self.groups = np.arange(1, self.p+1)
        self.ngroup = len(np.unique(self.groups))

        # handle kwargs
        for key in kwargs:
            self.kwargs[key] = kwargs[key]

        ## 1. Handle spline kwargs and create basis representation
        self.n_knots = self.kwargs.pop("n_knots", 1)
        self.degree = self.kwargs.pop("degree", 3)

        # Create knots for splines
        quantiles = np.linspace(
            1/(self.n_knots + 1), 
            self.n_knots / (self.n_knots + 1), 
            self.n_knots
        )
        self.knots = np.quantile(
            np.concatenate([self.X, self.Xk], axis=0),
            quantiles,
            axis=0
        ) # knots are p x num_knots

        # Create features
        # Todo: if we can reorder the basis functions so they are next to each other,
        # it may improve caching performance.
        self.features = []
        for tX in [self.X, self.Xk]:
            # spline basis representation: note `bases` has shape n x (degree*p)
            bases = np.concatenate([tX**j for j in range(1, self.degree+1)], axis=1)
            knotdiffs = tX.reshape(self.n, self.p, 1) - self.knots.reshape(1, self.p, self.n_knots)
            knotdiffs = np.maximum(knotdiffs, 0)**self.degree
            knotdiffs = np.concatenate(
                [knotdiffs[:, :, k] for k in range(self.n_knots)], 
                axis=1
            ) # reshape so this is n x (n_knots*p) 
            # combine
            bases = np.concatenate([bases, knotdiffs], axis=1)
            self.features.append(bases)
        self.features = np.concatenate(self.features, axis=1)

        # Create dummy groups showing which bases belong to which feature
        self.basis_groups = np.concatenate(
            [self.groups for _ in range(self.n_knots + self.degree)],
            axis=0
        )
        self.basis_group_sizes = utilities.calc_group_sizes(self.basis_groups).astype(int)
        self.basis_group_blocks = _calc_group_blocks(
            self.basis_groups, self.basis_group_sizes
        )
        self.n_basis_groups = len(self.basis_group_sizes)

        ## Posterior sampling
        self.n_iter = self.kwargs.pop("n_iter", 2000)
        self.chains = self.kwargs.pop("chains", 5)
        self.N = int(self.n_iter * self.chains)
        self.burn = int(self.kwargs.pop("burn_prop", 0.1) * self.n_iter)

        # Check whether this is binary or linear regression
        support = np.unique(self.y)
        if len(support) == 2:
            probit = int(1)
            z = (self.y == support[0]).astype(int)
        else:
            probit = int(0)
            z = np.zeros(self.n).astype(int)

        # Posterior sampling
        all_out = []
        for chain in range(self.chains):
            self.basis_groups -= 1
            out = _sample_mlr_spikeslab_group(
                N=self.n_iter + self.burn,
                features=self.features,
                groups=self.basis_groups,
                blocks=self.basis_group_blocks,
                gsizes=self.basis_group_sizes,
                max_gsize=np.max(self.basis_group_sizes).astype(int),
                y=self.y.astype(np.float64),
                z=z,
                probit=probit,
                # **self.kwargs
            )
            self.basis_groups += 1
            all_out.append(out)
        self.betas = np.concatenate([x['betas'][self.burn:] for x in all_out])
        self.etas = np.concatenate([x['etas'][self.burn:] for x in all_out])
        self.p0s = np.concatenate([x['p0s'][self.burn:] for x in all_out])
        self.psis = np.concatenate([x['psis'][self.burn:] for x in all_out])
        self.tau2s = np.concatenate([x['tau2s'][self.burn:] for x in all_out])
        self.sigma2s = np.concatenate([x['sigma2s'][self.burn:] for x in all_out])
        if probit == 1:
            self.y_latents = np.concatenate([x['y_latent'][self.burn:] for x in all_out])
        return self.compute_W()


class OracleMLR(MLR_Spikeslab):
    """
    Oracle masked likelihood ratio statistics for generalized additive models.

    Parameters
    ----------
    beta : np.ndarray
        ``p``-dimensional array of linear coefficients.
    """
    def __init__(self, beta, **kwargs):
        self.beta = beta
        self.kwargs = kwargs
        self.Z = None
        self.score = None
        self.score_type = None
        # Adjusted MLR stat, makes no difference for oracle though
        self.adjusted_mlr = self.kwargs.pop("adjusted_mlr", False)
        self.fdr = self.kwargs.pop("fdr", 0.05)

    def fit(self, X, Xk, groups, y, **kwargs):
        self.n, self.p = X.shape
        self.X = X
        self.Xk = Xk
        self.features = np.concatenate([X, Xk], axis=1)
        self.y = y
        self.groups = groups
        self.ungrouped = check_no_groups(self.groups, self.p, error='Groups not implemented for OracleMLR.')
        self.ngroup = self.p
        for key in kwargs:
            self.kwargs[key] = kwargs[key]

        # number of iterations to run
        self.n_iter = self.kwargs.pop("n_iter", 1000)
        self.chains = self.kwargs.pop("chains", 2)
        self.N = int(self.n_iter * self.chains)
        self.burn = int(self.kwargs.pop("burn_prop", 0.1) * self.n_iter)

        # Check whether we are in the binary setting
        support = np.unique(self.y)
        self.binary = len(support) == 2

        # Posterior sampling
        all_out = []
        for chain in range(self.chains):
            if self.binary:
                out = _sample_mlr_oracle_logistic(
                    N=self.n_iter + self.burn,
                    beta=self.beta,
                    features=self.features,
                    y=y.astype(np.float64),
                    **self.kwargs
                )
            else:
                out = _sample_mlr_oracle_gaussian(
                    N=self.n_iter + self.burn,
                    beta=self.beta,
                    features=self.features,
                    y=y.astype(np.float64),
                    **self.kwargs
                )
            all_out.append(out)
        self.etas = np.concatenate([x['etas'][self.burn:] for x in all_out])
        self.psis = np.concatenate([x['psis'][self.burn:] for x in all_out])
        self.betas = np.stack([self.beta for _ in range(len(self.psis))], axis=0)
        return self.compute_W()

class MLR_FX_Spikeslab(kstats.FeatureStatistic):
    """
    Masked likelihood ratio statistics using a spike-and-slab
    linear model. This is a specialized class designed to lead
    to slightly faster computation for fixed-X knockoffs.

    The arguments are the same as those for MLR_Spikeslab, 
    with the exception of the arguments listed below.

    Parameters
    ----------
    num_mixture : int
        Number of mixtures for the "slab" component of the 
        spike and slab. Defaults to 1.
    tau2 : float or list of floats
        Prior variance on nonzero coefficients. Default: 1.0.
    tau2_a0 : float or list of floats
        ``tau2`` has an InvGamma(``tau2_a0``, ``tau2_b0``) hyperprior. 
        When ``n_mixture`` > 1, this can be a list of length
        ``n_mixture``. Default: 2.0.
    tau2_b0 : float or list of floats
        ``tau2`` has an InvGamma(``tau2_a0``, ``tau2_b0``) hyperprior.
        When ``n_mixture`` > 1, this can be a list of length
        ``n_mixture``. Default: 0.01.
    """

    def __init__(
        self,
        **kwargs
    ):
        # Dummy attributes
        self.Z = None
        self.score = None
        self.score_type = None
        self.kwargs = kwargs

        # Defaults
        self.kwargs["n_iter"] = self.kwargs.get("n_iter", 2000)
        self.kwargs["burn_prop"] = self.kwargs.get("burn_prop", 0.1)
        self.kwargs["chains"] = self.kwargs.get("chains", 5)

        # adjusted MLR parameters
        self.adjusted_mlr = self.kwargs.pop("adjusted_mlr", False)
        self.fdr = self.kwargs.pop("fdr", 0.05)

    def calc_whiteout_statistics(
        self,
        X,
        Xk,
        y,
        S,
        calc_hatxi=True
    ):
        # Save inputs
        self.X = X
        self.Xk = Xk
        self.y = y
        self.diag_S = np.ascontiguousarray(np.diag(S))

        # Compute tildebeta, xi, hatxi, and so on
        Sinv = np.diag(1 / self.diag_S)
        self.tildebeta = np.dot(Sinv, np.dot(X.T - Xk.T, y))
        self.atb = np.abs(self.tildebeta.copy()) # atb = abs(tildebeta)
        self.A = np.dot(X.T, X) - S / 2 # covariance matrix of xi
        self.xi = np.dot(X.T + Xk.T, y) / 2
        if calc_hatxi:
            self.hatxi = scipy.linalg.solve(self.A, self.xi)

    def fit(
        self, X, Xk, groups, y, **kwargs
    ):

        # Save inputs and switch to whiteout framework
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.groups = groups
        self.ungrouped = check_no_groups(
            groups, 
            self.p, 
            error="The specialized FX MLR class does not support group knockoffs---use the generic class instead."
        )
        S = X.T @ X - X.T @ Xk
        self.calc_whiteout_statistics(X=X, Xk=Xk, y=y, S=S, calc_hatxi=False)
        #self.sigma2 = kstats.compute_residual_variance(X=X, Xk=Xk, y=y)
        self.XTX = np.dot(X.T, X)
        self.L = np.linalg.cholesky(self.A)
        self.Linv = np.ascontiguousarray(
            scipy.linalg.lapack.dtrtri(self.L, lower=True)[0]
        )
        self.Linv_xi = np.dot(self.Linv, self.xi)

        # Deal with kwargs
        for key in kwargs:
            self.kwargs[key] = kwargs[key]
        # handle mixture components and size of arrays
        self.num_mixture = self.kwargs.pop("num_mixture", 1)
        tau2_a0 = self.kwargs.pop("tau2_a0", 2.0)
        # Inverse-Gamma prior on tau2
        if isinstance(tau2_a0, float) or isinstance(tau2_a0, int):
            tau2_a0 = [tau2_a0 for _ in range(self.num_mixture)]
        self.tau2_a0 = np.array(tau2_a0, dtype=float)
        tau2_b0 = self.kwargs.pop("tau2_b0", 0.01)
        if isinstance(tau2_b0, float) or isinstance(tau2_b0, int):
            tau2_b0 = [tau2_b0 for _ in range(self.num_mixture)]
        self.tau2_b0 = np.array(tau2_b0, dtype=float)

        # kwargs that cannot be passed to the underlying cython
        self.n_iter = self.kwargs.pop("n_iter", 2000)
        self.chains = self.kwargs.pop("chains", 5)
        self.N = int(self.n_iter * self.chains)
        self.burn = int(self.kwargs.pop("burn_prop", 0.1) * self.n_iter)

        # Posterior sampling
        all_out = []
        for chain in range(self.chains):
            out = _sample_mlr_spikeslab_fx(
                N=self.n_iter + self.burn,
                xi=self.xi,
                atb=self.atb,
                XTX=self.XTX,
                diag_S=self.diag_S,
                num_mixture=self.num_mixture,
                tau2_a0s=self.tau2_a0,
                tau2_b0s=self.tau2_b0,
                A=self.A,
                L=self.L,
                Linv_xi=self.Linv_xi,
                **self.kwargs,
            )
            all_out.append(out)
        self.betas = np.concatenate([x['betas'][self.burn:] for x in all_out])
        self.etas = np.concatenate([x['etas'][self.burn:] for x in all_out])
        self.p0s = np.concatenate([x['p0s'][self.burn:] for x in all_out])
        self.tau2s = np.concatenate([x['tau2s'][self.burn:] for x in all_out])
        self.stb = np.concatenate([x['stb'][self.burn:] for x in all_out]) # sign(tildebeta)
        self.sigma2s = np.concatenate([x['sigma2'][self.burn:] for x in all_out])
        self.mixtures = np.concatenate([x['mixtures'][self.burn:] for x in all_out])
        return self.compute_W(signs=self.betas)

    def compute_W(self, signs=None):
        if signs is None:
            signs = self.betas

        # 1. Guess sign(beta)
        self.sign_guess = np.sign(
            np.sum(signs > 0, axis=0) - np.sum(signs < 0, axis=0)
        )
        nzeros = np.sum(self.sign_guess == 0)
        self.sign_guess[self.sign_guess == 0] = 1 - 2*np.random.binomial(1, 0.5, nzeros)
        self.wrong_guesses = np.sign(self.tildebeta) != self.sign_guess
        ## regular MLR statistics
        if not self.adjusted_mlr:			
            # 2. Compute log(P(tildebeta = sign_guess))
            eta_g0 = self.sign_guess == np.sign(signs)
            adj_eta = self.etas * (2*eta_g0 - 1)
            etas_cat = np.concatenate(
                [
                    adj_eta.reshape(self.N, self.p, 1), 
                    np.zeros((self.N, self.p, 1))
                ],
                axis=2
            )
            # This equals: log(P(sgn(tildebeta) = sign guess))
            log_prob = scipy.special.log_softmax(etas_cat, axis=2)[:, :, 0]
            log_prob = scipy.special.logsumexp(log_prob, b=1/self.N, axis=0)
            self.W = np.exp(log_prob) - 0.5

            # 3. Compute sign(W)
            self.W[self.wrong_guesses] = -1 * self.W[self.wrong_guesses]
            return self.W
        ## Adjusted MLR statistics
        else:
            tol = 1e-10
            # Compute P_j(MLR_j > 0 | D)
            mlr_pos = np.clip(np.mean(self.stb == self.sign_guess, axis=0), tol, 1-tol)
            # Compute P(MLR_j > 0, j is non-null | masked data)
            mlr_pos_nonnull = np.clip(
                np.mean((self.stb == self.sign_guess) * (self.betas != 0), axis=0), tol, 1-tol
            )
            # MLR signs
            mlr_sign = 1 - 2 * self.wrong_guesses
            # Return AMLR stats
            self.W = _mlr_to_adj_mlr(
                mlr_sign=mlr_sign, 
                prob_mlr_pos=mlr_pos, 
                prob_mlr_pos_nonnull=mlr_pos_nonnull, 
                fdr=self.fdr,
            )
            return self.W
