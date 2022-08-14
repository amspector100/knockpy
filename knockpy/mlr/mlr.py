"""
Note to self: should this be in "knockoff_stats.py"?
"""

import numpy as np
import scipy.special
from ._mlr_spikeslab_fx import _sample_mlr_spikeslab_fx
from ._mlr_spikeslab_mx import _sample_mlr_spikeslab_mx
from .. import knockoff_stats as kstats
from .. import utilities

def check_no_groups(groups, p):
	if groups is not None:
		if np.any(groups != np.arange(1, p+1)):
			raise ValueError(
				"This implementation of MLR stats. does not yet support group knockoffs."
			)


class MLR_MX_Spikeslab(kstats.FeatureStatistic):
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
    chain : int
    	Number of MCMC chains to run. Default: 5.
    burn_prop : float
    	The burn-in for each chain will be equal to 
    	``n_iter * burn_prop``.
	p0 : float
		Prior probability that any coefficient equals zero.
	update_p0 : bool
		If True, updates ``p0`` using a Beta hyperprior on ``p0``.
		Else, the value of ``p0`` is fixed. Default: True.
	p0_a0 : float
		If ``update_p0`` is True, ``p0`` has a
		Beta(``p0_a0``, ``p0_b0``, ``min_p0``) hyperprior.
		Default: 1.0.
	p0_b0 : float
		If ``update_p0`` is True, ``p0`` has a
		TruncBeta(``p0_a0``, ``p0_b0``, ``min_p0``) hyperprior.
		Default: 1.0.
	min_p0 : float
		Minimum value for ``p0`` as specified by the prior.
		Default: 0.5.
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
		Default: 0.01.
	tau2 : float
		Prior variance on nonzero coefficients. Default: 1.0.
	update_tau2 : bool
		If True, imposes an InverseGamma hyperprior on ``tau2``.
		Else, the value of ``tau2`` is fixed. Default: True.
	tau2_a0 : float
		If ``update_sigma2`` is True, ``tau2`` has an
		InvGamma(``tau2_a0``, ``tau2_b0``) hyperprior. 
		Default: 2.0.
	tau2_b0 : float
		If ``update_sigma2`` is True, ``tau2`` has an
		InvGamma(``tau2_a0``, ``tau2_b0``) hyperprior.
		Default: 0.01.

	Returns
	-------
	W : np.ndarray
		a ``p``-dimensional array of feature statistics.

	Notes
	-----
	Does not yet support group knockoffs. 
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

	def fit(
		self, X, Xk, y, groups, **kwargs
	):
		self.n = X.shape[0]
		self.p = X.shape[1]
		self.groups = groups
		check_no_groups(self.groups, self.p)
		# We do not yet support group knockoffs
		if self.groups is not None:
			if np.any(self.groups != np.arange(1, self.p+1)):
				raise ValueError(
					"This implementation of MLR stats. does not yet support group knockoffs."
				)

		self.features = np.concatenate([X, Xk], axis=1)
		for key in kwargs:
			self.kwargs[key] = kwargs[key]
		# kwargs that cannot be passed to the underlying cython
		self.n_iter = self.kwargs.pop("n_iter")
		self.chains = self.kwargs.pop("chains")
		self.N = int(self.n_iter * self.chains)
		self.burn = int(self.kwargs.pop("burn_prop", 0.1) * self.n_iter)

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
			out = _sample_mlr_spikeslab_mx(
				N=self.n_iter + self.burn,
				features=self.features,
				y=y.astype(np.float64),
				z=z,
				probit=probit,
				**self.kwargs
			)
			all_out.append(out)
		self.betas = np.concatenate([x['betas'][self.burn:] for x in all_out])
		self.etas = np.concatenate([x['etas'][self.burn:] for x in all_out])
		self.p0s = np.concatenate([x['p0s'][self.burn:] for x in all_out])
		self.psis = np.concatenate([x['psis'][self.burn:] for x in all_out])
		self.tau2s = np.concatenate([x['tau2s'][self.burn:] for x in all_out])
		self.sigma2s = np.concatenate([x['sigma2s'][self.burn:] for x in all_out])

		# Compute P(choose feature)
		# etas = log(P(choose feature) / P(choose knockoff))
		etas_cat = np.concatenate(
			[
				self.etas.reshape(self.N, self.p, 1),
				np.zeros((self.N, self.p, 1))
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

class MLR_FX_Spikeslab(kstats.FeatureStatistic):
	"""
	Masked likelihood ratio statistics using a spike-and-slab
	linear model. This is a specialized class designed to lead
	to slightly faster computation for fixed-X knockoffs.

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
    chain : int
    	Number of MCMC chains to run. Default: 5.
    burn_prop : float
    	The burn-in for each chain will be equal to 
    	``n_iter * burn_prop``.
    num_mixture : int
    	Number of mixtures for the "slab" component of the 
    	spike and slab. Defaults to 1.
	p0 : float
		Prior probability that any coefficient equals zero.
	update_p0 : bool
		If True, updates ``p0`` using a Beta hyperprior on ``p0``.
		Else, the value of ``p0`` is fixed. Default: True.
	p0_a0 : float
		If ``update_p0`` is True, ``p0`` has a
		Beta(``p0_a0``, ``p0_b0``, ``min_p0``) hyperprior.
		Default: 1.0.
	p0_b0 : float
		If ``update_p0`` is True, ``p0`` has a
		TruncBeta(``p0_a0``, ``p0_b0``, ``min_p0``) hyperprior.
		Default: 1.0.
	min_p0 : float
		Minimum value for ``p0`` as specified by the prior.
		Default: 0.5.
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
		Default: 0.01.
	tau2 : float or list of floats
		Prior variance on nonzero coefficients. Default: 1.0.
	update_tau2 : bool
		If True, imposes an InverseGamma hyperprior on ``tau2``.
		Else, the value of ``tau2`` is fixed. Default: True.
	tau2_a0 : float or list of floats
		If ``update_sigma2`` is True, ``tau2`` has an
		InvGamma(``tau2_a0``, ``tau2_b0``) hyperprior. 
		When ``n_mixture`` > 1, this can be a list of length
		``n_mixture``. Default: 2.0.
	tau2_b0 : float or list of floats
		If ``update_sigma2`` is True, ``tau2`` has an
		InvGamma(``tau2_a0``, ``tau2_b0``) hyperprior.
		When ``n_mixture`` > 1, this can be a list of length
		``n_mixture``. Default: 0.01.

	Notes
	-----
	For fixed-X knockoffs, this will give identical outputs
	to the MLR_MX_SpikeSlab class, but it may be slightly
	faster.
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
		check_no_groups(groups, self.p)
		S = X.T @ X - X.T @ Xk
		self.calc_whiteout_statistics(X=X, Xk=Xk, y=y, S=S, calc_hatxi=False)
		self.sigma2 = kstats.compute_residual_variance(X=X, Xk=Xk, y=y)
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
		tau2_a0 = self.kwargs.get("tau2_a0", 2.0)
		# Inverse-Gamma prior on tau2
		if isinstance(tau2_a0, float) or isinstance(tau2_a0, int):
			tau2_a0 = [tau2_a0 for _ in range(self.num_mixture)]
		self.tau2_a0 = np.array(tau2_a0, dtype=float)
		tau2_b0 = self.kwargs.get("tau2_b0", 0.01)
		if isinstance(tau2_b0, float) or isinstance(tau2_b0, int):
			tau2_b0 = [tau2_b0 for _ in range(self.num_mixture)]
		self.tau2_b0 = np.array(tau2_b0, dtype=float)

		# kwargs that cannot be passed to the underlying cython
		self.n_iter = self.kwargs.pop("n_iter")
		self.chains = self.kwargs.pop("chains")
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
		self.stb = np.concatenate([x['stb'][self.burn:] for x in all_out])
		self.sigma2s = np.concatenate([x['sigma2'][self.burn:] for x in all_out])
		self.mixtures = np.concatenate([x['mixtures'][self.burn:] for x in all_out])
		return self.compute_W(signs=self.betas)

	def compute_W(self, signs):

		# 1. Guess sign(beta)
		self.sign_guess = np.sign(
			np.sum(signs > 0, axis=0) - np.sum(signs < 0, axis=0)
		)
		nzeros = np.sum(self.sign_guess == 0)
		self.sign_guess[self.sign_guess == 0] = 1 - 2*np.random.binomial(1, 0.5, nzeros)			# Compute log(P(tildebeta) = sign_guess)
		
		# 2. Compute P(log(tildebeta) = sign_guess)
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
		log_prob = scipy.special.logsumexp(log_prob, b=1/self.n_iter, axis=0)
		self.W = np.exp(log_prob) - 0.5
		
		# 3. Compute sign(W)
		self.wrong_guesses = np.sign(self.tildebeta) != self.sign_guess
		self.W[self.wrong_guesses] = -1 * self.W[self.wrong_guesses]
		return self.W
