# cython: profile=False

import time 
cimport cython
import numpy as np
import scipy.stats
cimport numpy as np
from numpy cimport PyArray_ZEROS
import scipy.linalg
cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, exp, fabs, sqrt, fmin, fmax, erfc
from ._update_hparams import _update_hparams

# Blas commonly used parameters
cdef double zero = 0, one = 1, neg1 = -1
cdef int inc_0 = 0;
cdef int inc_1 = 1
cdef char* trans_t = 'T'
cdef char* trans_n = 'N'
cdef char* triang_u = 'U'
cdef char* triang_l = 'L'
cdef double M_SQRT1_2 = sqrt(0.5)
cdef float MAX_EXP = 20

cdef double log_sum_exp(u, v):
	cdef double m = fmax(u, v)
	return m + log(
		exp(fmin(u-m, MAX_EXP)) + exp(fmin(v-m, MAX_EXP))
	)

# Truncated normal sampling was a bottleneck, so 
# some custom samplers are here.
# Ref: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.26.6892
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double random_uniform():
	cdef double r = rand()
	return r / RAND_MAX

@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _expo_tn_sampler(double a):
	"""
	Samples Z \sim N(0,1) | Z \in [a,infty).
	"""
	cdef double rho
	cdef double u 
	cdef double y # ~ expo(a)
	while True:
		y = random_uniform()
		y = -1*log(y) / a
		rho = exp(-0.5*y*y)
		u = random_uniform()
		if u <= rho:
			return y + a

@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _norm_tn_sampler(double a):
	"""
	Samples Z \sim N(0,1) | Z \in [a,infty)
	"""
	cdef double z;
	while True:
		z = np.random.randn()
		if a >= 0:
			z = fabs(z) 
		if z >= a:
			return z

@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _sample_truncnorm_std(double a):
	"""
	Samples Z \sim N(0,1) | Z \in [a, \infty)
	efficiencitly.
	"""

	# constants from the paper
	cdef double t1 = 0.150
	if a >= t1:
		return _expo_tn_sampler(a)
	else:
		return _norm_tn_sampler(a)

@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double sample_truncnorm(
	double mean,
	double var,
	double b,
	int lower_interval
):
	"""
	if lower_interval == 1, samples from
	Z \sim N(mean, var) | Z \in (-infty, b)
	else, samples from 
	Z \sim N(mean, var) | Z \in (b, infty)
	"""
	scale = sqrt(var)
	cdef double a = (b - mean) / scale
	cdef double z;
	if lower_interval == 0:
		z = _sample_truncnorm_std(a)
	else:
		z = -1 * _sample_truncnorm_std(-1*a)
	return mean + scale * z


def _sample_mlr_spikeslab(
	int N,
	double[:, ::1] features,
	double[::1] y, # outcomes for linear regression, not used for probit
	long[::1] z, # censored outcomes for probit model, not used for lin reg
	int probit=1,
	double tau2=0.01, # variance of coefficients
	int update_tau2=1,
	double tau2_a0=2.0,
	double tau2_b0=0.01,
	double sigma2=1.0,
	int update_sigma2=1,
	double sigma2_a0=2.0,
	double sigma2_b0=0.01,
	double p0=0.9,
	int update_p0=1,
	double min_p0=0.5,
	double p0_a0=1.0,
	double p0_b0=1.0,
):
	"""
	features = [X, Xk]
	"""
	# Initialize outputs
	cdef:
		# Useful constants
		int n = int(features.shape[0])
		int p = int(features.shape[1] / 2)
		int i, it, j

		# Initialize outputs
		np.ndarray[long, ndim=1] inds = np.arange(p)
		np.ndarray[double, ndim=1] p0s_arr = np.zeros((N,))
		double[::1] p0s = p0s_arr
		np.ndarray[double, ndim=2] betas_arr = np.zeros((N, p))
		double[:, ::1] betas = betas_arr
		# psis[i,j] = 1 iff at iter i, we think fT[j+p] is the true feature j 
		np.ndarray[long, ndim=2] psis_arr = np.random.binomial(1, 0.5, size=(N, p)).astype(int)
		long[:, ::1] psis = psis_arr
		# etas[i,j] = log(P(psis[i,j] = 0) / P(psis[i,j] = 1))
		np.ndarray[double, ndim=2] etas_arr = np.zeros((N, p))
		double[:, ::1] etas = etas_arr
		# tau2s = signal size
		np.ndarray[double, ndim=1] tau2s_arr = np.zeros((N,))
		double[::1] tau2s = tau2s_arr
		# sigma2s array
		np.ndarray[double, ndim=1] sigma2s_arr = np.zeros((N,))
		double[::1] sigma2s = sigma2s_arr

		# Initialize latent variables Y (only used in probit regression)
		np.ndarray[double, ndim=2] Y_latent_arr = np.zeros((N, n))
		double[:, ::1] Y_latent = Y_latent_arr

		# Proposals, only used if min_p0 > 0
		int max_nprop = 100
		np.ndarray[double, ndim=1] p0_proposal_arr = np.zeros(max_nprop,)
		double[::1] p0_proposals = p0_proposal_arr

		# Precompute useful quantities 
		double[:, ::1] fT = np.ascontiguousarray(features.T)
		double[::1] l2norms = np.power(features, 2).sum(axis=0)
		double[::1] logdets = np.zeros((2*p, ))
		double logodds = log(p0) - log(1 - p0)
		double[::1] post_vars = np.zeros((2*p,))

		# scratch
		double negbetaj, old_betaj
		double XjTr, log_ratio_num, log_ratio_denom, log_det, ratio, r1s
		double[::1] post_means = np.zeros((2,))
		double[::1] kappas = np.zeros((2,))
		double[::1] log_psis = np.zeros((2,))
		double kappa, post_mean, log_psi, mup # mup = mu posterior
		double ratio_psi, kappa_psi, ratio_beta, kappa_beta, u, delta
		int num_active, offset

		# Initialize mu (predictions) and r (residuals)
		np.ndarray[double, ndim=1] mu_arr = np.zeros(n,)
		double[::1] mu = mu_arr
		np.ndarray[double, ndim=1] r_arr = np.zeros(n,)
		double[::1] r = r_arr


	# precompute sigma2 posterior variance
	cdef double sigma_a = n / 2.0 + sigma2_a0
	cdef np.ndarray [double, ndim=1] invgammas = scipy.stats.invgamma(sigma_a).rvs(N) 

	# initialize
	sigma2s[0] = sigma2
	tau2s[0] = tau2
	p0s[0] = p0

	# Initializations of latent variables (in probit case) and residuals
	if probit == 1:
		for it in range(n):
			Y_latent[0, it] = sample_truncnorm(
				mean=mu[it], 
				var=sigma2,
				b=0.0, 
				lower_interval=z[it] 
			)
			y[it] = Y_latent[0, it]
	for it in range(n):
		r[it] = y[it] - mu[it]

	for i in range(N):
		sigma2 = sigma2s[i]
		# precompute log determinants / posterior variances
		for j in range(2*p):
			logdets[j] = log(1 + tau2s[i] * l2norms[j] / sigma2) / 2
			post_vars[j] = 1 / (1 / tau2s[i] + l2norms[j] / sigma2)

		# ### Testing only
		# # Assert mu = np.dot(X, beta)
		# beta_ext = np.zeros(2*p)
		# for j in range(p):
		# 	if psis[i,j] == 0:
		# 		beta_ext[j] = betas[i,j]
		# 	else:
		# 		beta_ext[j+p] = betas[i,j]
		# mu2 = np.dot(features, beta_ext)
		# if probit == 1:
		# 	r2 = Y_latent[i] - mu_arr
		# else:
		# 	r2 = np.zeros(n)
		# 	for iii in range(n):
		# 		r2[iii] = y[iii] - mu2[iii]
		# print(f"mu - mu2 outer: {np.abs(mu_arr - mu2).mean()}")
		# print(f"r - r2 outer: {np.abs(r_arr - r2).mean()}")
		# ### end testing

		# Update beta and features
		np.random.shuffle(inds)
		for j in inds:
			# ### Testing only
			# # Assert mu = np.dot(X, beta)
			# if j == 0:
			# 	beta_ext = np.zeros(2*p)
			# 	for j in range(p):
			# 		if psis[i,j] == 0:
			# 			beta_ext[j] = betas[i,j]
			# 		else:
			# 			beta_ext[j+p] = betas[i,j]
			# 	mu2 = np.dot(features, beta_ext)
			# 	if probit == 1:
			# 		r2 = Y_latent[i] - mu_arr
			# 	else:
			# 		r2 = np.zeros(n)
			# 		for iii in range(n):
			# 			r2[iii] = y[iii] - mu2[iii]
			# 	print(f"mu - mu2 inner: {np.abs(mu_arr - mu2).mean()}")
			# 	print(f"r - r2 inner: {np.abs(r_arr - r2).mean()}")
			# 	### end testing

			# determine whether features[j] or features[j+p] was selected
			if psis[i,j] == 0:
				offset = 0
			else:
				offset = p
			# possibly reset residuals/mu to zero out betaj
			old_betaj = betas[i, j]
			if old_betaj != 0:
				blas.daxpy(&n, &old_betaj, &fT[j+offset,0], &inc_1, &r[0], &inc_1)
				delta = -1 * old_betaj
				blas.daxpy(&n, &delta, &fT[j+offset,0], &inc_1, &mu[0], &inc_1)

			# Resample which feature is the real feature
			for k in range(2):
				if k == 0:
					offset = 0
				else:
					offset = p
				# Compute post dist for betaj
				# 1. compute log ratio P(betaj = 0) / P(betaj != 0)
				XjTr = blas.ddot(
					&n,
					&r[0],
					&inc_1,
					&fT[j+offset, 0],
					&inc_1
				)
				# if j == 0:
				# 	print(f"offset={offset},XjTr={XjTr}, Xjoffset0={fT[j+offset, 0]}")
				log_ratio_num = tau2s[i] * XjTr * XjTr / sigma2
				log_ratio_denom = 2 * (sigma2 + tau2s[i] * l2norms[j+offset])
				logratio = logodds - log_ratio_num / log_ratio_denom + logdets[j+offset]
				ratio = exp(fmin(logratio, MAX_EXP))
				kappa = ratio / (1.0 + ratio)
				# 2. Calculate posterior mean condition on betaj ! = 0
				mup = post_vars[j+offset] * XjTr / sigma2
				# Calculate log(P(psi[i,j] = k))
				# rank one scale
				r1s = post_vars[j+offset] / sigma2 / (sigma2 + post_vars[j+offset] * l2norms[j+offset])
				# the (r - mu Xj)^T cov^{-1} (r - mu) term
				log_psi = (-2.0*mup*XjTr + mup*mup*l2norms[j+offset]) / sigma2
				log_psi -= r1s * (XjTr*XjTr - 2.0*mup*XjTr*l2norms[j+offset])
				log_psi -= r1s * (mup*mup*l2norms[j+offset]*l2norms[j+offset])
				log_psi = -1*log_psi / 2
				# Add log((1-kappa) * logdet)
				log_psi -= log(1 + post_vars[j+offset] * l2norms[j+offset] / sigma2) / 2
				log_psi += log(1 - kappa)
				log_psi = log_sum_exp(log(kappa), log_psi)
				# Save these to arrays
				log_psis[k] = log_psi
				kappas[k] = kappa
				post_means[k] = mup

			### Actually calculate log odds and reset psi
			etas[i, j] = log_psis[0] - log_psis[1]
			ratio_psi = exp(fmin(etas[i, j], MAX_EXP))
			kappa_psi = ratio_psi / (1.0 + ratio_psi)

			### debugging
			# if j == 0:
			# 	print(f"j={j}, eta={etas[i,j]}, kappas={kappas[0], kappas[1]}, mus={post_means[0], post_means[1]}")

			u = random_uniform()
			if u <= kappa_psi:
				psis[i,j] = 0
				offset = 0
				post_mean = post_means[0]
				kappa_beta = kappas[0]
			else:
				psis[i,j] = 1
				offset = p
				post_mean = post_means[1]
				kappa_beta = kappas[1]

			### Reset betaj
			u = random_uniform()
			if u <= kappa_beta:
				betas[i, j] = 0
			else:
				betas[i, j] = np.sqrt(post_vars[j+offset]) * np.random.randn() + post_mean

			# Update Z, r, mu
			if betas[i, j] != 0:
				# update mu
				blas.daxpy(
					&n,
					&betas[i,j],
					&fT[j+offset,0],
					&inc_1,
					&mu[0],
					&inc_1
				)
				# update r. For probit, this is redundant
				# since we will update r after updating Y_latent anyway
				if probit == 0:
					negbetaj = -1 * betas[i,j]
					blas.daxpy(
						&n, 
						&negbetaj, 
						&fT[j+offset,0], 
						&inc_1, 
						&r[0], 
						&inc_1
					)

			# Special case for probit: truncated normal sampling
			if probit == 1 and (betas[i, j] != 0 or old_betaj != 0):
				# make sure sgn(z) matches y
				for it in range(n):
					Y_latent[i, it] = sample_truncnorm(
						mean=mu[it], var=sigma2, b=0.0, lower_interval=z[it] 
					)
					r[it] = Y_latent[i, it] - mu[it]

		# Update hyperparams
		_update_hparams(
			i=i, 
			n=n,
			p=p,
			max_nprop=max_nprop,
			betas=betas,
			r=r,
			p0s=p0s,
			sigma2s=sigma2s,
			tau2s=tau2s,
			invgamma=invgammas[i],
			p0_proposals=p0_proposals,
			update_tau2=update_tau2,
			tau2_a0=tau2_a0,
			tau2_b0=tau2_b0,
			update_sigma2=update_sigma2,
			sigma2_a0=sigma2_a0,
			sigma2_b0=sigma2_b0,
			update_p0=update_p0,
			min_p0=min_p0,
			p0_a0=p0_a0,
			p0_b0=p0_b0,
		)

		# Recompute logodds
		logodds = log(p0s[i]) - log(1 - p0s[i])

		# Set new betas, p0s to be old values
		if i != N - 1:
			betas[i+1] = betas[i]
			p0s[i+1] = p0s[i]
			sigma2s[i+1] = sigma2s[i]
			tau2s[i+1] = tau2s[i]
			psis[i+1] = psis[i]
			etas[i+1] = etas[i]

	output = {
		"betas":betas_arr,
		"p0s":p0s_arr,
		"etas":etas_arr,
		"psis":psis_arr,
		"tau2s":tau2s_arr,
		"sigma2s":sigma2s_arr,
	}
	if probit == 1:
		output['y_latent'] = Y_latent_arr
	return output