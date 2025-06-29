# cython: profile=True

import time 
cimport cython
import numpy as np
import scipy.stats
cimport numpy as np
from numpy cimport PyArray_ZEROS
import scipy.linalg
import scipy.special
cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, exp, fabs, sqrt, fmin, fmax, erfc

# Blas commonly used parameters
cdef double zero = 0.0, one = 1.0, neg1 = -1.0
cdef int inc_0 = 0
cdef int inc_1 = 1
cdef char* trans_t = 'T'
cdef char* trans_n = 'N'
cdef char* triang_u = 'U'
cdef char* triang_l = 'L'
cdef float MAX_EXP = 20

@cython.nonecheck(False)
@cython.cdivision(True)
cdef double random_uniform():
	cdef double r = rand()
	return r / RAND_MAX

cdef int _weighted_choice(
	double[::1] probs, # dummy variables, just for memory
	double[::1] log_probs,
	int num_mixture
):
	cdef int ii
	cdef double max_log_prob = 0.0
	cdef double denom = 0.0
	cdef double cumsum = 0.0
	# Avoid overflow
	for ii in range(num_mixture + 1):
		max_log_prob = fmax(log_probs[ii], max_log_prob)
	# Compute probabilities
	for ii in range(num_mixture + 1):
		probs[ii] = exp(fmin(MAX_EXP, log_probs[ii] - max_log_prob))
		denom += probs[ii]
	# Normalize
	for ii in range(num_mixture + 1):
		probs[ii] = probs[ii] / denom
	# Choose integer
	cdef double u = random_uniform()
	for ii in range(num_mixture + 1):
		cumsum += probs[ii]
		if cumsum >= u:
			return ii

def _sample_mlr_spikeslab_fx(
	int N,
	double[::1] xi,
	double[::1] atb,
	double[:, ::1] XTX,
	double[::1] diag_S,
	double[:, ::1] A,
	double[:, ::1] L,
	double[::1] Linv_xi,
	double[::1] tau2_a0s,
	double[::1] tau2_b0s,
	int num_mixture, # must be >= 1
	double sigma2=1.0,
	int update_p0=1,
	double p0_a0=1.0,
	double p0_b0=1.0,
	double sigma2_a0=2.0,
	double sigma2_b0=0.01,
	int update_sigma2=1,
):
	# Note: atb = abs(tildebeta), stb=sign(tildebeta)
	# vartb = var(tildebeta)
	# Initialize outputs
	cdef:
		# Useful constants
		int p = atb.shape[0]
		int i, it, j, k

		# Initialize outputs
		np.ndarray[long, ndim=1] inds = np.arange(p)

		# sparsity parameter
		np.ndarray[double, ndim=2] p0s_arr = np.zeros((N, num_mixture + 1,))
		double[:, ::1] p0s = p0s_arr
		# array to hold updates
		p0s_update_arr = np.zeros((num_mixture + 1,))
		#double[::1] p0s_update = p0s_update_arr 
		# Track dirichlet params for sparsity
		np.ndarray[double, ndim=1] d_params_arr = np.zeros((num_mixture + 1,))
		double[::1] d_params = d_params_arr

		# coefficients
		np.ndarray[double, ndim=2] betas_arr = np.zeros((N, p))
		double[:, ::1] betas = betas_arr
		# log-odds
		np.ndarray[double, ndim=2] etas_arr = np.zeros((N, p))
		double[:, ::1] etas = etas_arr
		# sign(tildebeta)
		np.ndarray[double, ndim=2] stb_arr = np.zeros((N, p))
		double[:, ::1] stb = stb_arr
		# tau2 for various components
		np.ndarray[double, ndim=2] tau2s_arr = np.zeros((N, num_mixture))
		double[:, ::1] tau2s = tau2s_arr
		
		# track mixture components
		np.ndarray[long, ndim=2] mixtures_arr = np.zeros((N,p), dtype=np.int64)
		long[:, ::1] mixtures = mixtures_arr
		# Track logits / probs for each component
		np.ndarray[double, ndim=1] m_logits_arr = np.ones((num_mixture + 1,))
		double[::1] mixture_logits = m_logits_arr
		np.ndarray[double, ndim=1] mixture_probs_arr = np.zeros((num_mixture + 1,))
		double[::1] mixture_probs = mixture_probs_arr
		
		# sigma2s array
		np.ndarray[double, ndim=1] sigma2s_arr = np.zeros((N,))
		double[::1] sigma2s = sigma2s_arr

		# scratch
		double u, ratio, kappa, temp, r2
		int num_active
		double XTXjbeta, cent_ols_dot
		double rank_one_scale, log_diff_exp, log_diff_det
		double post_mean, post_var
		double sample_var
		double term1, term2, sigma2_a, sigma2_b

		# tildebeta, vartb, just for convenience
		np.ndarray[double, ndim=1] tb_arr = np.zeros((p,))
		double[::1] tb = tb_arr
		np.ndarray[double, ndim=1] scratch_arr = np.zeros((p,))
		double[::1] scratch = scratch_arr

	### DEBUGGING ONLY: DELETE LATER
	# Siginv = np.linalg.inv(XTX)
	# cdef np.ndarray[double, ndim=1] mixture_logits_debug
	# cdef np.ndarray[double, ndim=2] ols_dots = np.zeros((N, p))

	# precompute sigma2 posterior variance
	sigma2_a = p + sigma2_a0
	cdef np.ndarray [double, ndim=1] gammas = scipy.stats.gamma(a=sigma2_a).rvs(N) 

	# For debugging only---delete later
	# cdef np.ndarray[double, ndim=2] L_arr = np.zeros((p, p))
	# for i in range(p):
	# 	for j in range(p):
	# 		L_arr[i,j] = L[i,j]

	# intialize tb, tau2, d_params and p0s
	sigma2s[0] = sigma2
	for j in range(p):
		tb[j] = atb[j] * stb[0, j]
	for k in range(num_mixture):
		tau2s[0, k] = tau2_b0s[k] / scipy.stats.gamma(a=tau2_a0s[k]).rvs()
	# Create dirichlet prior parameters
	d_params[0] = p0_a0
	for k in range(num_mixture):
		d_params[k+1] = p0_b0 / num_mixture
	# initialize p0s
	if update_p0 == 1:
		p0s_update_arr = np.random.dirichlet(alpha=d_params_arr)
		for k in range(num_mixture + 1):
			p0s[0, k] = p0s_update_arr[k]
	else:
		for k in range(num_mixture + 1):
			p0s[0, k] = d_params[k] / (p0_a0 + p0_b0)

	for i in range(N):
		np.random.shuffle(inds)
		mixture_logits[0] = log(p0s[i, 0])
		for j in inds:
			# Step 1: Calculate mixture component | tildebeta, hatxi, beta_{-j}
			# zero out betaj for computation
			betas[i,j] = 0

			# Useful precmputation for all tau2s
			# this is equivalent to dot(ols - beta, XTX[j])
			XTXjbeta = blas.ddot(
				&p,
				&XTX[j,0],
				&inc_1,
				&betas[i,0],
				&inc_1
			)
			cent_ols_dot = xi[j] + tb[j] * diag_S[j] / 2 - XTXjbeta

			# # DEBUGGING seems like it works
			# if j == 0:
			# 	ols_dots[i, j] = cent_ols_dot
			# 	hatbeta = Siginv @ xi + 1/2 * Siginv @ (np.array(diag_S) * tb_arr)
			# 	cent_hatbeta = hatbeta - betas_arr[i]
			# 	cent_ols_dot_v2 = XTX[j] @ cent_hatbeta
			# 	print("HERE", np.abs(cent_ols_dot - cent_ols_dot_v2))


			for k in range(num_mixture):
				### 1a. Calculate ratio of determinants of covs
				# This is O(1)
				# Using matrix determinant lemma:
				# det(XTXinv + tau2s[i, k] ej ejT)
				# = det(XTXinv) (1 + tau2s[i, k] ejT XTX ej)
				log_diff_det = (-1/2) * log(
					1 + tau2s[i, k] * XTX[j, j] / sigma2s[i]
				)

				# 1b. Using rank-one theory, log of ratio of exps in pdf
				# First, calculate rank-one scaling constant in O(1)
				rank_one_scale = tau2s[i, k] / (sigma2s[i]**2) / (
					1 + tau2s[i, k] * XTX[j, j] / sigma2s[i]
				)
				log_diff_exp = (1/2) * rank_one_scale * cent_ols_dot * cent_ols_dot

				# 1c. Compute logit and cache
				mixture_logits[k+1] = log(p0s[i,k+1]) + log_diff_exp + log_diff_det

			#### Debugging only
			# mixture_logits_debug = np.zeros(num_mixture + 1)
			# mixture_logits_debug[0] = log(p0s[i,0])
			# hatbeta = Siginv @ xi + 1/2 * Siginv @ (np.array(diag_S) * tb_arr)
			# cent_hatbeta = hatbeta - betas_arr[i]
			# cent_ols_dot_v2 = XTX[j] @ cent_hatbeta
			# for k in range(1, num_mixture+1):
			# 	mixture_logits_debug[k] = log(p0s[i, k])
			# 	mixture_logits_debug[k] -= 1/2 * np.log(1 + tau2s[i, k-1] / sigma2s[i] * XTX[j, j])
			# 	mixture_logits_debug[k] += tau2s[i, k-1] * cent_ols_dot_v2**2 / (
			# 		2 * (sigma2s[i]**2) * (1 + tau2s[i, k-1] / sigma2s[i] * XTX[j, j])
			# 	)

			#print("HERE", np.abs(mixture_logits_debug - m_logits_arr).mean()) 
			k = _weighted_choice(
				probs=mixture_probs,
				log_probs=mixture_logits,
				num_mixture=num_mixture,
			)
			### Debugging only
			#mixture_logits_debug = scipy.special.softmax(mixture_logits_debug)
			#print(m_logits_arr)
			#print(mixture_probs_arr)
			#print(mixture_logits_debug)
			#print("HERE2", f"k={k}", np.abs(mixture_probs - mixture_logits_debug).mean())
			#print(k, mixture_logits[0], mixture_logits[1])#, mixture_logits[2])
			mixtures[i, j] = k # record this came from mixture k
			if k == 0:
				betas[i, j] = 0
			else:
				k -= 1 # change index when working with tau2s
				rank_one_scale = tau2s[i, k] / (sigma2s[i]**2) / (
					1 + tau2s[i, k] * XTX[j, j] / sigma2s[i]
				)
				temp = tau2s[i, k] * (1 / sigma2s[i] - rank_one_scale * XTX[j,j])
				cond_mean = temp * cent_ols_dot
				cond_var = tau2s[i, k] * (1 - temp * XTX[j,j])
				# 2c. Sample
				cond_var = fmax(0, cond_var)
				#print(f"cond_var={cond_var}, cond_mean={cond_mean}")
				betas[i,j] = np.sqrt(cond_var) * np.random.randn() + cond_mean

			if betas[i, j] != 0:
				etas[i,j] = fabs(betas[i,j] * atb[j]) * diag_S[j]
				ratio = exp(fmin(etas[i,j], MAX_EXP))
				kappa = ratio / (1.0 + ratio)
			else:
				etas[i,j] = 0
				kappa = 0.5
			# Resample stb and reset tb
			u = random_uniform()
			######## CHECK THIS VERY CAREFULLY
			if u <= kappa: # then sgn(stb[j]) == sgn(betas[i,j])
				stb[i,j] = 1
			else:
				stb[i,j] = -1
			if betas[i,j] < 0:
				stb[i,j] = -1 * stb[i,j]
			tb[j] = stb[i,j] * atb[j]

		# Resample p0s, start by finding number of variables
		# per component
		d_params[0] = p0_a0
		for k in range(num_mixture):
			d_params[k+1] = p0_b0 / num_mixture
		for j in range(p):
			d_params[mixtures[i, j]] += 1

		# sample p0
		if update_p0 == 1:
			p0s_update_arr = np.random.dirichlet(
				alpha=d_params_arr
			)
			for k in range(num_mixture + 1):
				p0s[i, k] = p0s_update_arr[k]

		# Resample tau2
		for k in range(num_mixture):
			sample_var = 0
			num_active = 0
			for j in range(p):
				if mixtures[i, j] == k + 1:
					num_active += 1
					sample_var += betas[i,j] * betas[i,j]
			tau2s[i, k] = (tau2_b0s[k] + sample_var / 2.0) / np.random.gamma(
			 	shape=tau2_a0s[k] + float(num_active) / 2.0
			)
			#print(f"i={i}, k={k}, tau2s[i, k] = {tau2s[i,k]}")

		# Resample sigma2s
		if update_sigma2 == 1:
			# 1. Calculate ||S^{1/2} (tilde{beta} - beta)||_2^2
			for j in range(p):
				scratch[j] = tb[j] - betas[i,j]
				scratch[j] = scratch[j] * sqrt(diag_S[j])

			term1 = blas.dnrm2(&p, &scratch[0], &inc_1)
			term1 = term1 * term1
			# 2. scratch = - L^T beta
			blas.dgemm(
				trans_n, # transA
				trans_n, # transB, doesn't matter (B is p x 1)
				&p, # M = op(A).shape[0]
				&inc_1, # N = op(B).shape[1]
				&p, # K = op(A).shape[1] = op(B).shape[0]
				&neg1, # alpha
				&L[0,0], # A
				&p, # LDA first dim of A in calling program
				&betas[i,0], # B
				&p, # LDB first dim of B in calling program
				&zero, # beta
				&scratch[0], # C
				&p, # first dim of C in calling program
			)
			# scratch = Linv xi - L^{-1/2} beta
			blas.daxpy(
				&p, # dimensionality
				&one, # alpha
				&Linv_xi[0], # vector to add
				&inc_1, # spacing
				&scratch[0], # output vector
				&inc_1, # spacing
			)
			# term2 = ||scratch||_2^2
			#print(scratch_arr)
			term2 = blas.dnrm2(&p, &scratch[0], &inc_1)
			term2 = term2 * term2
			# resample sigma2
			sigma2_b = (term1 / 2.0 + term2) / 2.0 + sigma2_b0
			sigma2s[i] = sigma2_b / gammas[i]

		# Set new betas, p0s to be old values (temporarily)
		if i != N - 1:
			betas[i+1] = betas[i]
			stb[i+1] = stb[i]
			sigma2s[i+1] = sigma2s[i]
			for k in range(num_mixture + 1):
				p0s[i+1, k] = p0s[i, k]
			for k in range(num_mixture):
				tau2s[i+1, k] = tau2s[i, k]

	return {
		"betas":betas_arr,
		"etas":etas_arr,
		"p0s":p0s_arr,
		"tau2s":tau2s_arr,
		"stb":stb_arr,
		"sigma2":sigma2s_arr,
		"mixtures":mixtures_arr,
	}