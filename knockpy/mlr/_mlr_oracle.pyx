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

# Truncated normal sampling was a bottleneck, so 
# some custom samplers are here.
# Ref: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.26.6892
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double random_uniform():
	cdef double r = rand()
	return r / RAND_MAX

def _sample_mlr_oracle_gaussian(
	int N,
	double[:, ::1] features,
	double[::1] beta, # coefficients
	double[::1] y, # outcomes for linear regression, not used for probit
	#long[::1] z, # censored outcomes for probit model, not used for lin reg
	double sigma2=1,
):
	"""
	features = [X, Xk]
	"""
	# Initialize outputs
	cdef:
		# Useful constants
		int n = int(features.shape[0])
		int p = int(features.shape[1] / 2)
		int i, it, j, ii, jj, iii, jjj

		# Initialize outputs
		np.ndarray[long, ndim=1] inds = np.arange(p)
		# psis[i,j] = 1 iff at iter i, we think fT[j+p] is the true feature j 
		np.ndarray[long, ndim=2] psis_arr = np.random.binomial(1, 0.5, size=(N, p)).astype(int)
		long[:, ::1] psis = psis_arr
		# etas[i,j] = log(P(psis[i,j] = 0) / P(psis[i,j] = 1))
		np.ndarray[double, ndim=2] etas_arr = np.zeros((N, p))
		double[:, ::1] etas = etas_arr

		# Precompute useful quantities 
		double[:, ::1] fT = np.ascontiguousarray(features.T)
		double[::1] l2norms = np.power(features, 2).sum(axis=0)

		# scratch
		double XjTr, tXjTr, log_ratio, ratio, kappa, u, delta
		double log_rat_num, log_rat_denom
		int offset

		# Initialize mu (predictions) and r (residuals)
		np.ndarray[double, ndim=1] r_arr = np.zeros(n,)
		double[::1] r = r_arr

	# initialize r
	for it in range(n):
		r[it] = y[it]
	cdef np.ndarray[double, ndim=1] beta_ext_arr = np.zeros(2*p,)
	cdef double[::1] beta_ext = beta_ext_arr
	for j in range(p):
		beta_ext[j] = (1-psis[0,j]) * beta[j]
		beta_ext[j+p] = psis[0,j] * beta[j]
	r_arr -= np.dot(features, beta_ext)

	for i in range(N):
		# #### TESTING ONLY ###
		# for j in range(p):
		# 	beta_ext[j] = (1-psis[i,j]) * beta[j]
		# 	beta_ext[j+p] = psis[i,j] * beta[j]
		# mu2 = np.dot(features, beta_ext)
		# r2 = np.zeros(n)
		# for it in range(n):
		# 	r2[it] = y[it] - mu2[it]
		# print(f"diff btwn r2, r is {np.abs(r2 - r).mean()}")
		### End testing

		# Update features
		np.random.shuffle(inds)
		for j in inds:
			# Ignore null inds, where we just sample randomly
			if beta[j] == 0:
				etas[i,j] = 0
				u = random_uniform()
				if u <= 0.5:
					psis[i,j] = 0
					offset = 0
				else:
					psis[i,j] = p
					offset = p
			else:
				# for non-null inds, reset residuals; first determine whether features[j] or features[j+p] was selected
				if psis[i,j] == 0:
					offset = 0
				else:
					offset = p
				# reset residuals and mu
				blas.daxpy(&n, &beta[j], &fT[j+offset,0], &inc_1, &r[0], &inc_1)

				# Compute log-odds of which is the feature vs. knockoff
				XjTr = blas.ddot(&n, &r[0], &inc_1, &fT[j, 0], &inc_1)
				tXjTr = blas.ddot(&n, &r[0], &inc_1, &fT[j+p, 0], &inc_1)
				log_rat_num = beta[j] * XjTr - beta[j] * beta[j] * l2norms[j] / 2
				log_rat_denom = beta[j] * tXjTr - beta[j] * beta[j] * l2norms[j+p] / 2
				log_ratio = (log_rat_num - log_rat_denom) / sigma2
				etas[i,j] = log_ratio

				# Pick feature vs. knockoff
				ratio = exp(fmin(log_ratio, MAX_EXP))
				kappa = ratio / (1.0 + ratio)
				u = random_uniform()
				if u <= kappa:
					psis[i,j] = 0
					offset = 0
				else:
					psis[i,j] = 1
					offset = p

				delta = -1 * beta[j]
				blas.daxpy(
					&n, 
					&delta, 
					&fT[j+offset,0], 
					&inc_1, 
					&r[0], 
					&inc_1
				)

		# Set new psis, etas to be old values
		if i != N - 1:
			psis[i+1] = psis[i]
			etas[i+1] = etas[i]

	output = {
		"etas":etas_arr,
		"psis":psis_arr,
	}
	return output


def _sample_mlr_oracle_logistic(
	int N,
	double[:, ::1] features,
	double[::1] beta, # coefficients
	double[::1] y, # logistic reg. outcomes 
):
	"""
	features = [X, Xk]
	"""
	# Initialize outputs
	cdef:
		# Useful constants
		int n = int(features.shape[0])
		int p = int(features.shape[1] / 2)
		int i, it, j, ii, jj, iii, jjj, offset, k

		# Initialize outputs
		np.ndarray[long, ndim=1] inds = np.arange(p)
		# psis[i,j] = 1 iff at iter i, we think fT[j+p] is the true feature j 
		np.ndarray[long, ndim=2] psis_arr = np.random.binomial(1, 0.5, size=(N, p)).astype(int)
		long[:, ::1] psis = psis_arr
		# etas[i,j] = log(P(psis[i,j] = 0) / P(psis[i,j] = 1))
		np.ndarray[double, ndim=2] etas_arr = np.zeros((N, p))
		double[:, ::1] etas = etas_arr

		# Precompute useful quantities 
		double[:, ::1] fT = np.ascontiguousarray(features.T)

		# scratch
		double log_ratio, ratio, kappa, u, delta
		double y_log_odds_i
		double[::1] lls = np.zeros(2,)

		# Initialize mu and predictions
		np.ndarray[double, ndim=1] mu_arr = np.zeros(n,)
		double[::1] mu = mu_arr

	# Initialize mu
	cdef np.ndarray[double, ndim=1] beta_ext_arr = np.zeros(2*p,)
	cdef double[::1] beta_ext = beta_ext_arr
	for j in range(p):
		beta_ext[j] = (1-psis[0,j]) * beta[j]
		beta_ext[j+p] = psis[0,j] * beta[j]
	mu_arr += np.dot(features, beta_ext)

	for i in range(N):
		# Update features
		np.random.shuffle(inds)
		for j in inds:
			
			# #### TESTING ONLY ###
			# for k in range(p):
			# 	beta_ext[k] = (1-psis[i,k]) * beta[k]
			# 	beta_ext[k+p] = psis[i,k] * beta[k]
			# mu2 = np.dot(features, beta_ext)
			# print(f"at i={i}, j={j}, mudiff={np.abs(mu2 - mu_arr).mean()}")
			# ## End testing

			# Ignore null inds, where we just sample randomly
			if beta[j] == 0:
				etas[i,j] = 0
				u = random_uniform()
				if u <= 0.5:
					psis[i,j] = 0
					offset = 0
				else:
					psis[i,j] = p
					offset = p
			else:
				# for non-null inds, reset mu; first determine whether features[j] or features[j+p] was selected
				if psis[i,j] == 0:
					offset = 0
				else:
					offset = p
				# reset residuals and mu
				delta = -1 * beta[j]

				blas.daxpy(&n, &delta, &fT[j+offset,0], &inc_1, &mu[0], &inc_1)

				# Compute log-odds of which is the feature vs. knockoff
				# by comparing the log-likelihood for the feature vs. knockoff
				lls[0] = 0 # ll for feature
				lls[1] = 0 # ll for knockoff
				for it in range(2):
					if it == 0:
						offset = 0
					else:
						offset = p						
					for ii in range(n):
						y_log_odds_i = mu[ii] + beta[j] * fT[j+offset,ii] # ln(p(y=1) / p(y=0))
						lls[it] += y[ii] * y_log_odds_i 
						lls[it] -= log(1 + exp(y_log_odds_i))
				log_ratio = lls[0] - lls[1] 
				etas[i,j] = log_ratio

				# Pick feature vs. knockoff
				ratio = exp(fmin(log_ratio, MAX_EXP))
				kappa = ratio / (1.0 + ratio)
				u = random_uniform()
				if u <= kappa:
					psis[i,j] = 0
					offset = 0
				else:
					psis[i,j] = 1
					offset = p
				# Update mu
				blas.daxpy(&n, &beta[j], &fT[j+offset,0], &inc_1, &mu[0], &inc_1)


		# Set new psis, etas to be old values
		if i != N - 1:
			psis[i+1] = psis[i]
			etas[i+1] = etas[i]

	output = {
		"etas":etas_arr,
		"psis":psis_arr,
	}
	return output