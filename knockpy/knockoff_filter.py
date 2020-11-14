import numpy as np
from . import utilities
from . import mrc
from .knockoffs import GaussianSampler
from . import metro
from . import knockoff_stats as kstats 

class KnockoffFilter:
	"""
	:param fixedX: If True, creates fixed-X knockoffs.
	Defaults to False (model-X knockoffs).
	"""
	def __init__(self, fixedX=False):

		# Initialize some flags / options
		self.debias = False
		self.fixedX = fixedX
		if self.debias and self.fixedX:
			raise ValueError(
				"Debiased lasso not yet implemented for FX knockoffs"
			)
		self._sdp_degen = None
		self.S = None

	def sample_knockoffs(self):

		# SDP degen flag (for internal use)
		if self._sdp_degen is None:
			if "_sdp_degen" in self.knockoff_kwargs:
				self._sdp_degen = self.knockoff_kwargs.pop("_sdp_degen")
			else:
				self._sdp_degen = False

		# If fixedX, signal this to knockoff kwargs
		if self.fixedX:
			self.knockoff_kwargs['fixedX'] = True
			Sigma = None
		else:
			Sigma = self.Sigma

		# If we have already computed S, signal this
		# because this is expensive
		if self.S is not None:
			if 'S' not in self.knockoff_kwargs:
				self.knockoff_kwargs['S'] = self.S

		# Initial sample from Gaussian
		if self.knockoff_type == 'gaussian':
			knockoffs, S = gaussian_knockoffs(
				X=self.X, 
				groups=self.groups,
				mu=self.mu,
				Sigma=Sigma,
				return_S=True,
				**self.knockoff_kwargs,
			)
			knockoffs = knockoffs[:, :, 0]
		# Alternatively sample from ARTK
		elif self.knockoff_type == 'artk':
			# Sample
			self.knockoff_sampler = metro.ARTKSampler(
				X=self.X,
				V=self.Sigma,
				**self.knockoff_kwargs,
			)
			knockoffs = self.knockoff_sampler.sample_knockoffs()

			# Extract S
			inv_order = self.knockoff_sampler.inv_order
			S = self.knockoff_sampler.S[inv_order][:, inv_order]
		# Or block T metro
		elif self.knockoff_type == 'blockt':
			# Sample
			self.knockoff_sampler = metro.BlockTSampler(
				X=self.X,
				V=self.Sigma,
				**self.knockoff_kwargs,
			)
			knockoffs = self.knockoff_sampler.sample_knockoffs()

			# Extract S
			S = self.knockoff_sampler.S
		elif self.knockoff_type == 'ising':
			if 'gibbs_graph' not in self.knockoff_kwargs:
				raise IndexError(
					f"For ising knockoffs, must provide gibbs graph as knockoff_kwarg"
				)
			self.knockoff_sampler = metro.IsingKnockoffSampler(
				X=self.X,
				V=self.Sigma,
				mu=self.mu,
				**self.knockoff_kwargs
			)
			knockoffs = self.knockoff_sampler.sample_knockoffs()

			# It is difficult to extract S analytically 
			# here because there are different S's for
			# different parts of the data
			S = None

		else:
			raise ValueError(
				f"knockoff_type must be one of 'gaussian', 'artk', 'ising', 'blockt', not {knockoff_type}"
			)

		# Possibly use recycling
		if self.recycle_up_to is not None:

			# Split
			rec_knockoffs = self.X[:self.recycle_up_to]
			new_knockoffs = knockoffs[self.recycle_up_to:]

			# Combine
			knockoffs = np.concatenate((rec_knockoffs, new_knockoffs), axis=0)

		# For high precision simulations of degenerate knockoffs,
		# ensure degeneracy
		if self._sdp_degen:
			sumcols = self.X[:, 0] + knockoffs[:, 0]
			knockoffs = sumcols.reshape(-1, 1) - self.X

		self.knockoffs = knockoffs
		self.S = S

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
			# Only case we don't invert Ginv is when G has been
			# explicitly constructed to be low rank
			if not self._sdp_degen:
				self.Ginv = utilities.chol2inv(self.G)
			else:
				self.Ginv = None
		else:
			self.G, self.Ginv = utilities.estimate_covariance(
				np.concatenate([self.X, self.knockoffs], axis=1)
			)

		return knockoffs

	def make_selections(self, W, fdr):
		"""" Calculate data dependent threshhold and selections """
		T = kstats.data_dependent_threshhold(W=W, fdr=fdr)
		selected_flags = (W >= T).astype("float32")
		return selected_flags

	def forward(
		self,
		X,
		y,
		mu=None,
		Sigma=None,
		groups=None,
		knockoffs=None,
		feature_stat="lasso",
		fdr=0.10,
		feature_stat_kwargs={},
		knockoff_type='gaussian',
		knockoff_kwargs={"sdp_verbose": False},
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
		:param shrinkage
		:param recycle_up_to: Three options:
			- if None, does nothing.
			- if an integer > 1, uses the first "recycle_up_to"
			rows of X as the the first "recycle_up_to" rows of knockoffs.
			- if a float between 0 and 1 (inclusive), interpreted
			as the proportion of knockoffs to recycle. 
		For more on recycling, see https://arxiv.org/abs/1602.03574
		"""

		# Preliminaries - infer covariance matrix for MX 
		if Sigma is None and not self.fixedX:
			if 'sdp_tol' in knockoff_kwargs:
				tol = knockoff_kwargs['sdp_tol']
			else:
				tol = 1e-2
			Sigma, _ = utilities.estimate_covariance(X, tol, shrinkage)
		feature_stat_kwargs = feature_stat_kwargs.copy()

		# Save objects
		self.X = X
		self.mu = mu
		self.Sigma = Sigma
		self.groups = groups
		self.knockoff_kwargs = knockoff_kwargs
		self.knockoff_type = str(knockoff_type).lower()

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
 
		# Parse feature statistic function
		if feature_stat == "lasso":
			feature_stat = kstats.LassoStatistic()
			if 'debias' in feature_stat_kwargs:
				if feature_stat_kwargs['debias']:
					self.debias = True
		elif feature_stat == 'dlasso':
			feature_stat = kstats.LassoStatistic()
			self.debias = True
		elif feature_stat == 'ridge':
			feature_stat = kstats.RidgeStatistic()
		elif feature_stat == "ols":
			feature_stat = kstats.OLSStatistic()
		elif feature_stat == "margcorr":
			feature_stat = kstats.MargCorrStatistic()
		elif feature_stat == 'randomforest':
			feature_stat = kstats.RandomForestStatistic()
		elif feature_stat == 'deeppink':
			feature_stat = kstats.DeepPinkStatistic()

		# Sample knockoffs
		if knockoffs is None:
			knockoffs = self.sample_knockoffs()
			if self.debias:
				# This is only computed if self.debias is True
				feature_stat_kwargs["Ginv"] = self.Ginv
				feature_stat_kwargs['debias'] = True

		# Feature statistics
		feature_stat.fit(
			X=X, knockoffs=knockoffs, y=y, groups=groups, **feature_stat_kwargs
		)
		# Inherit some attributes
		self.fstat = feature_stat
		self.Z = self.fstat.Z
		self.W = self.fstat.W
		self.score = self.fstat.score
		self.score_type = self.fstat.score_type

		self.selected_flags = self.make_selections(self.W, fdr)

		# Return
		return self.selected_flags