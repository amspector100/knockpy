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
from ._mlr_spikeslab import sample_truncnorm

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

@cython.nonecheck(False)
@cython.cdivision(True)
cdef double log_sum_exp(u, v):
	cdef double m = fmax(u, v)
	return m + log(exp(u - m) + exp(v - m))

@cython.nonecheck(False)
@cython.cdivision(True)
cdef double random_uniform():
	cdef double r = rand()
	return r / RAND_MAX

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int _zero_phiT(
	double[:, ::1] phiT,
	int gsize,
	int n
):
	"""
	Fills XAT with zeros.
	"""
	cdef int nb = n * gsize
	blas.daxpy(
		&nb,
		&neg1,
		&phiT[0,0],
		&inc_1,
		&phiT[0,0],
		&inc_1
	)
	return 0

cdef double _compute_suff_stats_and_Qj(
	double[:, ::1] fT,
	double[:, ::1] phiT,
	double[:, ::1] fTf,
	double[:, ::1] phiTphi,
	double[::1] r,
	double[::1] phiTr,
	double[::1] mu_post,
	double[:, ::1] Qj,
	long[:, ::1] blocks,
	int gj, # the group we are interested in
	int n,
	int offset,
	int max_gsize,
	int gsize,
	double tau2, 
	double sigma2,
):
	"""
	Computes phiTphi and phiTr for the full model.
	These are the only O(n) operations required
	in the full loop, 
	"""
	cdef int ii, jj, jj2, j, j2, INFO, b2
	cdef int g2 = max_gsize * max_gsize

	# 1. Loop through fill phi. Note
	# phi is initialized to all zeros.
	ii = 0
	for jj in range(gsize):
		j = blocks[gj, jj]
		if j < 0:
			raise ValueError(f"error: at gj={gj}, jj={jj}, j={j}<0")
		# Set phi[:, ii] = features[:, j]
		blas.daxpy(
			&n,
			&one,
			&fT[j+offset,0],
			&inc_1,
			&phiT[ii,0],
			&inc_1
		)
		ii += 1

	# Step 2. Set phiTphi = np.dot(phi.T, phi)
	for jj in range(gsize):
		j = blocks[gj, jj]
		if j < 0:
			raise ValueError(f"error: at gj={gj}, jj={jj}, j={j}<0")
		for jj2 in range(gsize):
			j2 = blocks[gj, jj2]
			assert j2 >= 0
			phiTphi[jj, jj2] = fTf[j+offset, j2+offset]

	# blas.dgemm(
	# 	trans_t, # transA
	# 	trans_n, # transB
	# 	&max_gsize, # M = op(A).shape[0]
	# 	&max_gsize, # N = op(B).shape[1]
	# 	&n, # K = op(A).shape[1] = op(B).shape[0]
	# 	&one, # alpha
	# 	&phiT[0,0], # A
	# 	&n, # LDA first dim of A in calling program
	# 	&phiT[0,0], # B
	# 	&n, # LDB first dim of B in calling program
	# 	&zero, # beta
	# 	&phiTphi[0,0], # C
	# 	&max_gsize, # first dim of C in calling program
	# )

	# Step 3. Set phiTr = np.dot(phi, r)
	blas.dgemm(
		trans_t, # transA
		trans_n, # transB
		&max_gsize, # M = op(A).shape[0]
		&inc_1, # N = op(B).shape[1]
		&n, # K = op(A).shape[1] = op(B).shape[0]
		&one, # alpha
		&phiT[0,0], # A
		&n, # LDA first dim of A in calling program
		&r[0], # B
		&n, # LDB first dim of B in calling program
		&zero, # beta
		&phiTr[0], # C
		&max_gsize, # first dim of C in calling program
	)

	# Step 4. Set Qj = I + tau2 / sigma2 phiT phi
	# start by setting Qj = identity
	for ii in range(gsize):
		for jj in range(gsize):
			if ii == jj:
				Qj[ii, jj] = 1.0
			else:
				Qj[ii, jj] = 0.0
	# Set QA = I + tau2 / sigma2 phiTphi
	cdef double cm_scale = tau2 / sigma2
	blas.daxpy(
		&g2,
		&cm_scale,
		&phiTphi[0,0],
		&inc_1,
		&Qj[0,0],
		&inc_1,
	)

	# Step 5. Qj = cholesky decomp of Qj
	lapack.dpotrf(
		triang_l, # UPLO, upper vs. lower triangle of A
		&gsize, # dim of matrix
		&Qj[0, 0], # matrix to perform cholesky decomp on
		&max_gsize, # LDA = leading dim of QA in calling program
		&INFO # error output
	)
	if INFO != 0:
		raise RuntimeError(f"dpotrf for Qj exited with INFO={INFO}.")

	# Calculate log determinant
	cdef double log_det_Qj = 0.0
	for jj in range(gsize):
		log_det_Qj += 2.0 * log(Qj[jj, jj])

	# Step 6. Set Qj = inverse of itself
	lapack.dtrtri(
		triang_l, # UPLO, upper vs. lower triangular
		trans_n, # diag, n means not diagonal
		&gsize, # N dimension
		&Qj[0,0], # A
		&max_gsize, # LDA = leading dim of QA in calling program
		&INFO # error output
	)
	if INFO != 0:
		raise RuntimeError(f"dtrtri for Qj exited with INFO={INFO}.")

	# Step 7. Set mu_post = np.dot(Qj, phiTr)
	for jj in range(gsize):
		mu_post[jj] = phiTr[jj]
	blas.dtrmm(
		triang_l, # 'L' (left: op(A) @ B) or 'R' (right: B @ op(A))
		triang_l, # upper or lower for triang matrix
		trans_n, # transA, transpose A or not (I think no)
		trans_n, # diag, n means not unit triangular
		&gsize, # M = B.shape[0]
		&inc_1, # N = op(B).shape[1]
		&one, # alpha
		&Qj[0,0], # A
		&max_gsize, # LDA first dim of A in calling program
		&mu_post[0], # B
		&max_gsize, # LDB first dim of B in calling program
	)

	return log_det_Qj



def _sample_mlr_spikeslab_group(
	int N,
	double[:, ::1] features,
	long[::1] groups, # groups for group-sparsity prior
	long[:, ::1] blocks, # maps features to set of features in that group
	long[::1] gsizes, # size of each group,
	int max_gsize, # maximum group size, useful for allocating memory
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
	See Spector and Fithian (2022) for description of the algorithm.

	Note that this internal function breaks with knockpy convention and assumes 
	groups starts from zero and counts upwards.
	"""
	# Initialize outputs
	cdef:
		# Useful constants
		int n = int(features.shape[0])
		int d = int(features.shape[1] / 2)
		int p = np.unique(groups).shape[0]
		int i, it, j, ii, jj, iii, jjj, INFO, k
		long il, itl, jl # just to case these to longs 

		# Initialize outputs
		np.ndarray[long, ndim=1] inds = np.arange(p)
		np.ndarray[double, ndim=1] p0s_arr = np.zeros((N,))
		double[::1] p0s = p0s_arr
		np.ndarray[double, ndim=2] betas_arr = np.zeros((N, d))
		double[:, ::1] betas = betas_arr
		# psis[i,j] = 1 iff at iter i, we think fT[groupj+p] corresponds to the true features 
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
		np.ndarray[double, ndim=2] fT_arr = np.ascontiguousarray(features.T) 
		double[:, ::1] fT = fT_arr
		double[:, ::1] fTf = np.dot(features.T, features)
		#double[::1] l2norms = np.power(features, 2).sum(axis=0)
		double logodds = log(p0) - log(1 - p0)

		# scratch
		double negbetaj, old_betaj
		double log_ratio, log_det, ratio, r1s, cm_scale
		double[::1] kappas = np.zeros((2,))
		double[::1] log_psis = np.zeros((2,))
		double kappa, post_mean, log_psi
		double ratio_psi, kappa_psi, ratio_beta, kappa_beta, u, delta
		int num_active, offset, gsize

		# Initialize mu (predictions) and r (residuals)
		np.ndarray[double, ndim=1] mu_arr = np.zeros(n,)
		double[::1] mu = mu_arr
		np.ndarray[double, ndim=1] r_arr = np.zeros(n,)
		double[::1] r = r_arr

		# Inner products of phi_j (the set of all features associated with group j)
		# with itself and y.
		np.ndarray[double, ndim=2] phiT_arr = np.zeros((max_gsize, n))
		double[:, ::1] phiT = phiT_arr
		# np.ndarray[double, ndim=2] phiTphi_c = np.zeros((max_gsize, max_gsize))
		# double[:, ::1] phiTphi_cache = phiTphi_c
		np.ndarray[double, ndim=2] phiTphi_arr = np.zeros((max_gsize, max_gsize))
		double[:, ::1] phiTphi = phiTphi_arr
		# np.ndarray[double, ndim=1] phiTr_c = np.zeros(max_gsize)
		# double[::1] phiTr_cache = phiTr_c
		np.ndarray[double, ndim=1] phiTr_arr = np.zeros(max_gsize)
		double[::1] phiTr = phiTr_arr

		# matrix Qj from paper
		double log_det_Qj
		np.ndarray[double, ndim=2] Qj_arr = np.zeros((max_gsize, max_gsize))
		double[:, ::1] Qj = Qj_arr
		# The conditional mean / variance / related vectors
		np.ndarray[double, ndim=1] mu_post_arr = np.zeros(max_gsize,)
		double[::1] mu_post = mu_post_arr
		double[::1] mu_post_copy = np.zeros(max_gsize,) # because there's no in-place dgemm
		np.ndarray[double, ndim=2] V_arr = np.zeros((max_gsize, max_gsize)) # conditional covariance
		double[:, ::1] V = V_arr
		np.ndarray[double, ndim=2] Vs_arr = np.zeros((max_gsize, max_gsize)) # scratch for V
		double[:, ::1] Vs = Vs_arr # just scratch
		# Store updates for beta as contiguous array
		np.ndarray[double, ndim=1] beta_next_arr = np.zeros(max_gsize,)
		double[::1] beta_next = beta_next_arr


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

		# Update beta and features
		np.random.shuffle(inds)
		for j in inds:
			gsize = gsizes[j]
			# ### Testing only
			# # Assert mu = np.dot(X, beta)
			# if j == 0:
			# 	beta_ext = np.zeros(2*d)
			# 	for jj in range(p):
			# 		for ii in range(gsizes[jj]):
			# 			jjj = blocks[jj, ii]
			# 			assert jjj >= 0
			# 			if psis[i,jj] == 0:
			# 				beta_ext[jjj] = betas[i,jjj]
			# 			else:
			# 				beta_ext[jjj+d] = betas[i,jjj]
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

			# 1. Reset residuals
			# determine whether features[j] or features[j+p] was selected
			if psis[i,j] == 0:
				offset = 0
			else:
				offset = d
			# reset residuals/mu to zero out betaj. Do this for each
			# feature in the group.
			for ii in range(gsize):
				jj = blocks[j, ii]
				assert jj >= 0
				old_betaj = betas[i, jj]
				if old_betaj != 0:
					blas.daxpy(&n, &old_betaj, &fT[jj+offset,0], &inc_1, &r[0], &inc_1)
					delta = -1 * old_betaj
					blas.daxpy(&n, &delta, &fT[jj+offset,0], &inc_1, &mu[0], &inc_1)

			# 2. Resample which feature is the real feature
			for k in range(2):
				if k == 0:
					offset = 0
				else:
					offset = d
				
				# Compute sufficient statistics and Qj
				log_det_Qj = _compute_suff_stats_and_Qj(
					fT=fT,
					phiT=phiT,
					fTf=fTf,
					phiTphi=phiTphi,
					r=r,
					phiTr=phiTr,
					mu_post=mu_post,
					Qj=Qj,
					blocks=blocks,
					gj=j,
					n=n,
					offset=offset,
					max_gsize=max_gsize,
					gsize=gsize,
					tau2=tau2s[i],
					sigma2=sigma2s[i],
				)

				# ### Debugging only
				# groups_conf = np.array(groups)
				# ginds = np.where(groups_conf == j)[0] + offset
				# phiT_conf = fT_arr[ginds.astype(int)]
				# #print("phiT_conf - phiT", np.abs(phiT_conf - phiT_arr[0:len(ginds)]).mean())
				# phiTr_conf = phiT_conf @ r_arr
				# #print("phiTr_conf - phiTr", np.abs(phiTr_conf - phiTr_arr[0:len(ginds)]).mean())
				# phiTphi_conf = phiT_conf @ phiT_conf.T
				# phiTphi_guess = phiTphi_arr[0:gsize][:, 0:gsize]
				# #print("phiTphi diff:", np.abs(phiTphi_conf - phiTphi_guess).mean())
				# Qj_conf = np.eye(gsize) + tau2s[i] / sigma2s[i] * phiTphi_conf
				# Qj_chol_conf = np.linalg.cholesky(Qj_conf)
				# Qj_cholinv = np.linalg.inv(Qj_chol_conf)
				# # This doesn't work because Qjcholinv is filled with garbage but looks good based
				# # upon inspection
				# # Qj_cholinv_guess = Qj_arr[0:gsize][:, 0:gsize]
				# # print(Qj_cholinv, Qj_cholinv_guess, phiTphi_conf)
				# # print("Qj cholinv diff:", np.abs(Qj_cholinv_guess - Qj_cholinv).mean())
				# mu_post_conf = Qj_cholinv @ phiTr_conf
				# #print("mu_post diff:", np.abs(mu_post_conf - mu_post_arr[0:gsize]).mean())
				# ### End debugging

				# Zero out phiT to prepare for the next iteration
				_zero_phiT(phiT=phiT, gsize=gsize, n=n)

				# 1. compute log P(beta^{(j)} = 0) / P(beta^{(j)} != 0)
				exp_term = blas.dnrm2(&gsize, &mu_post[0], &inc_1)
				exp_term = exp_term * exp_term * tau2s[i] / (2*sigma2s[i]*sigma2s[i])
				log_ratio = logodds + log_det_Qj / 2.0 - exp_term
				ratio = exp(fmin(log_ratio, MAX_EXP))
				kappa = ratio / (1.0 + ratio)

				# ### Debugging only
				# exp_term_conf = phiTr_conf.T @ np.linalg.inv(Qj_conf) @ phiTr_conf
				# exp_term_conf *= tau2s[i] / (2 * sigma2s[i] * sigma2s[i])
				# print("exp term diff:", np.abs(exp_term_conf - exp_term), "/", np.around(exp_term, 5))
				# ### End debugging

				# 2. log P(Xj = this one | D), aka log psi
				log_psi = exp_term - log_det_Qj / 2.0 + log(1-kappa)
				log_psi = log_sum_exp(log(kappa), log_psi)

				# Save to arrays
				log_psis[k] = log_psi
				kappas[k] = kappa

			# Actually calculate log odds and reset psi
			etas[i, j] = log_psis[0] - log_psis[1]
			ratio_psi = exp(fmin(etas[i, j], MAX_EXP))
			kappa_psi = ratio_psi / (1.0 + ratio_psi)

			# ## debugging
			# if j == 0:
			# 	print(f"j={j}, eta={etas[i,j]}, kappas={kappas[0], kappas[1]}")

			u = random_uniform()
			if u <= kappa_psi:
				psis[i,j] = 0
				offset = 0
				kappa_beta = kappas[0]
			else:
				psis[i,j] = 1
				offset = d
				kappa_beta = kappas[1]

			# Reset betas
			u = random_uniform()
			if u <= kappa_beta:
				for ii in range(gsize):
					beta_next[ii] = 0.0
			else:
				# need to recompute sufficient statistics---caching could perhaps 
				# give a ~30% speedup here, but we would need to cache a lot
				log_det_Qj = _compute_suff_stats_and_Qj(
					fT=fT,
					phiT=phiT,
					fTf=fTf,
					phiTphi=phiTphi,
					r=r,
					phiTr=phiTr,
					mu_post=mu_post,
					Qj=Qj,
					blocks=blocks,
					gj=j,
					n=n,
					offset=offset,
					max_gsize=max_gsize,
					gsize=gsize,
					tau2=tau2s[i],
					sigma2=sigma2s[i],
				)

				#### Compute conditional mean
				# At this point, mu_post = Qj^{-1/2} phiTr
				# 1. multiply by Qj^{-1/2} again
				# so mu_post = Qj^{-1} phi^T r
				blas.dtrmm(
					triang_l, # 'L' (left: op(A) @ B) or 'R' (right: B @ op(A))
					triang_l, # upper or lower for triang matrix
					trans_t, # transA, transpose A or not
					trans_n, # diag, n means not unit triangular
					&gsize, # M = B.shape[0]
					&inc_1, # N = op(B).shape[1]
					&one, # alpha
					&Qj[0,0], # A
					&max_gsize, # LDA first dim of A in calling program
					&mu_post[0], # B
					&max_gsize, # LDB first dim of B in calling program
				)
				# ### Debugging only
				# QA_guess = tau2s[i] / sigma2s[i] * np.dot(phiT_arr, phiT_arr.T) + np.eye(max_gsize)
				# QA_guess = QA_guess[0:gsize][:, 0:gsize]
				# QAI = np.linalg.inv(QA_guess)
				# expected = np.dot(QAI, phiTr_arr[0:gsize])
				# print(f"H1 mu-expected={mu_post_arr[0:gsize] - expected}, mu={mu_post_arr[0:gsize]}, expected={expected}")
				# ### End debugging
				# 2. Multiply by tau2 / sigma2 * phiTphi
				# so mu_post =  tau^2 / sigma^2 * phi^T phi Qj^{-1} phi^T r
				# note mu_post_copy is used bc there is no in-place dgemm
				cm_scale = tau2s[i] / sigma2s[i]
				for ii in range(gsize):
					mu_post_copy[ii] = mu_post[ii]
				blas.dgemm(
					trans_n, # transA
					trans_n, # transB
					&gsize, # M = op(A).shape[0]
					&inc_1, # N = op(B).shape[1]
					&gsize, # K = op(A).shape[1] = op(B).shape[0]
					&cm_scale, # alpha
					&phiTphi[0,0], # A
					&max_gsize, # LDA first dim of A in calling program
					&mu_post_copy[0], # B
					&max_gsize, # LDB first dim of B in calling program
					&zero, # beta
					&mu_post[0], # C
					&max_gsize, # LDC first dim of C in calling program
				)
				# ### Debugging only
				# QA_guess = tau2s[i] / sigma2s[i] * np.dot(phiT_arr, phiT_arr.T) + np.eye(max_gsize)
				# QA_guess = QA_guess[0:gsize][:, 0:gsize]
				# QAI = np.linalg.inv(QA_guess)
				# expected = phiTphi_arr[0:gsize][:, 0:gsize] @ QAI @ phiTr_arr[0:gsize]
				# expected = tau2s[i] / sigma2s[i] * expected
				# print(f"H2 mu-expected={mu_post_arr[0:gsize] - expected}, mu={mu_post_arr[0:gsize]}, expected={expected}")
				# ### End debugging

				# 3. Set mu_post -= phiT r
				blas.daxpy(
					&gsize,
					&neg1,
					&phiTr[0],
					&inc_1,
					&mu_post[0],
					&inc_1
				)
				# finish conditional mean by multiplying by -tau2 / sigma2
				for ii in range(gsize):
					mu_post[ii] *= -1 * tau2s[i] / sigma2s[i]

				## DEBUGGING START ###
				# sigma2 = sigma2s[i]
				# tau2 = tau2s[i]
				# XA = phiT_arr[0:gsize].T
				# k = gsize
				# Sigma11 = sigma2 * np.eye(n) + tau2 * np.dot(XA, XA.T)
				# Sigma12 = tau2 * XA
				# Sigma22 = tau2 * np.eye(k)
				# Sigma = np.concatenate(
				# 	[np.concatenate([Sigma11, Sigma12], axis=1),
				# 	np.concatenate([Sigma12.T, Sigma22], axis=1)],
				# 	axis=0
				# )
				# expected = np.dot(np.dot(Sigma12.T, np.linalg.inv(Sigma11)), r_arr)
				# QA_guess = tau2s[i] / sigma2s[i] * np.dot(phiT_arr, phiT_arr.T) + np.eye(max_gsize)
				# QA_guess = QA_guess[0:gsize][:, 0:gsize]
				# QAI = np.linalg.inv(QA_guess)
				# t1 = tau2s[i] * phiTr_arr[0:gsize] / sigma2s[i]
				# t2 = np.dot(
				# 	phiTphi_arr[0:gsize][:, 0:gsize],
				# 	np.dot(QAI, phiTr_arr[0:gsize])
				# )
				# expected = t1 - ((tau2s[i] / sigma2s[i])**2) * t2
				# print(f"H3 mu-expected={mu_post_arr[0:gsize] - expected}, mu={mu_post_arr[0:gsize]}, expected={expected}")
				### DEBUGGING END ###

				# Now compute posterior covariance
				# 0. Set V = phiTphi
				for ii in range(gsize):
					for jj in range(gsize):
						V[ii, jj] = phiTphi[ii, jj]
				# print(f"C0: diff={np.abs(phiTphi_arr[0:gsize][:, 0:gsize] - V_arr[0:gsize][:, 0:gsize]).mean()}")

				# 1. Set V = LA^{-1} phiTphi
				blas.dtrmm(
					triang_l, # 'L' (left: op(A) @ B) or 'R' (right: B @ op(A))
					triang_l, # upper or lower for triang matrix
					trans_n, # transA, transpose A or not (I think no)
					trans_n, # diag, n means not unit triangular
					&gsize, # M = B.shape[0]
					&gsize, # N = op(B).shape[1]
					&one, # alpha
					&Qj[0,0], # A
					&max_gsize, # LDA first dim of A in calling program
					&V[0,0], # B
					&max_gsize, # LDB first dim of B in calling program
				)

				# 2. Set V = QA^{-1} phiTphi
				blas.dtrmm(
					triang_l, # 'L' (left: op(A) @ B) or 'R' (right: B @ op(A))
					triang_l, # upper or lower for triang matrix
					trans_t, # transA, transpose A or not (I think no)
					trans_n, # diag, n means not unit triangular
					&gsize, # M = B.shape[0]
					&gsize, # N = op(B).shape[1]
					&one, # alpha
					&Qj[0,0], # A
					&max_gsize, # LDA first dim of A in calling program
					&V[0,0], # B
					&max_gsize, # LDB first dim of B in calling program
				)
				# # DEBUGGING delete later
				# QA_guess = tau2s[i] / sigma2s[i] * np.dot(phiT_arr, phiT_arr.T) + np.eye(max_gsize)
				# QA_guess = QA_guess[0:gsize][:, 0:gsize]
				# QAI = np.linalg.inv(QA_guess)
				# expected = np.dot(QAI, phiTphi_arr[0:gsize][:, 0:gsize])
				# result = V_arr[0:gsize][:, 0:gsize]
				# print(f"C2 Vs-expected=\n{result - expected}\n Vs={result}, expected={expected}")

				# 3. Set Vs = phiTphi QA^{-1} phiTphi
				blas.dgemm(
					trans_n, # transA
					trans_n, # transB
					&gsize, # M = op(A).shape[0]
					&gsize, # N = op(B).shape[1]
					&gsize, # K = op(A).shape[1] = op(B).shape[0]
					&one, # alpha
					&phiTphi[0,0], # A
					&max_gsize, # LDA first dim of A in calling program
					&V[0,0], # B
					&max_gsize, # LDB first dim of B in calling program
					&zero, # beta
					&Vs[0,0], # C
					&max_gsize, # first dim of C in calling program
				)

				# 4. Set V to be the conditional variance:
				# tau2 I_{|A|} - tau2/sigma4 phiTphi + tau4/sigma6 phiTphi QA^{-1} phiTphi
				b2 = max_gsize * max_gsize
				blas.daxpy(
					&b2,
					&neg1,
					&V[0,0],
					&inc_1,
					&V[0,0],
					&inc_1,
				)
				for ii in range(gsize):
					V[ii, ii] = tau2s[i]
				cm_scale = -1 * tau2s[i] * tau2s[i] / sigma2s[i]
				blas.daxpy(
					&b2,
					&cm_scale,
					&phiTphi[0,0],
					&inc_1,
					&V[0,0],
					&inc_1,
				)
				cm_scale = tau2s[i] * tau2s[i] * tau2s[i] / (sigma2s[i] * sigma2s[i])
				blas.daxpy(
					&b2,
					&cm_scale,
					&Vs[0,0],
					&inc_1,
					&V[0,0],
					&inc_1,
				)
				# ### DEBUGGING
				# sigma2 = sigma2s[i]
				# tau2 = tau2s[i]
				# XA = phiT_arr[0:gsize].T
				# k = gsize
				# Sigma11 = sigma2 * np.eye(n) + tau2 * np.dot(XA, XA.T)
				# Sigma12 = tau2 * XA
				# Sigma22 = tau2 * np.eye(k)
				# Sigma = np.concatenate(
				# 	[np.concatenate([Sigma11, Sigma12], axis=1),
				# 	np.concatenate([Sigma12.T, Sigma22], axis=1)],
				# 	axis=0
				# )
				# expected = Sigma22 - np.dot(np.dot(Sigma12.T, np.linalg.inv(Sigma11)), Sigma12)
				# result = V_arr[0:gsize][:, 0:gsize]
				# print(f"C4 V-expected=\n{result - expected}")
				# ### END DEBUGGING

				# 5. Set V to be cholesky decomp of cond. variance for cholesky sampling
				lapack.dpotrf(
					triang_l, # UPLO, upper vs. lower triangle of A
					&gsize, # dim of matrix
					&V[0, 0], # matrix to perform cholesky decomp on
					&max_gsize, # LDA = leading dim of QA in calling program
					&INFO # error output
				)
				### FOR DEBUGGING DELETE LATER
				if INFO != 0:
					raise RuntimeError(f"dpotrf exited with INFO={INFO}. Try setting max_gsize=1.")

				### Sample i.i.d. std normals
				for jj in range(gsize):
					beta_next[jj] = np.random.randn()

				# ### DEBUGGING
				# beta_ns = beta_next.copy()

				# Apply V to beta_next
				blas.dtrmm(
					triang_l, # 'L' (left: op(A) @ B) or 'R' (right: B @ op(A))
					triang_l, # upper or lower for triang matrix
					trans_n, # transA, transpose A or not (I think no)
					trans_n, # diag, n means not unit triangular
					&gsize, # M = B.shape[0]
					&inc_1, # N = op(B).shape[1]
					&one, # alpha
					&V[0,0], # A
					&max_gsize, # LDA first dim of A in calling program
					&beta_next[0], # B
					&max_gsize, # LDB first dim of B in calling program
				)

				# Add mu
				blas.daxpy(
					&gsize,
					&one,
					&mu_post[0],
					&inc_1,
					&beta_next[0],
					&inc_1
				)

				# ### DEBUGGING ONLY 
				# sigma2 = sigma2s[i]
				# tau2 = tau2s[i]
				# XA = phiT_arr[0:gsize].T
				# k = gsize
				# Sigma11 = sigma2 * np.eye(n) + tau2 * np.dot(XA, XA.T)
				# Sigma12 = tau2 * XA
				# Sigma22 = tau2 * np.eye(k)
				# Sigma = np.concatenate(
				# 	[np.concatenate([Sigma11, Sigma12], axis=1),
				# 	np.concatenate([Sigma12.T, Sigma22], axis=1)],
				# 	axis=0
				# )
				# V_exp = Sigma22 - np.dot(np.dot(Sigma12.T, np.linalg.inv(Sigma11)), Sigma12)
				# mu_exp = np.dot(np.dot(Sigma12.T, np.linalg.inv(Sigma11)), r_arr)
				# expected = np.linalg.cholesky(V_exp) @ beta_ns[0:gsize] + mu_exp
				# result = beta_next_arr[0:gsize]
				# if j == 0:
				# 	print(f"Final check: result-expected, {np.abs(result-expected).mean()}")#, result={result}, expected={expected}")
				# ### END DEBUGGING

				# Important: zero out phiT for the next use
				_zero_phiT(phiT=phiT, gsize=gsize, n=n)

			# Update beta
			for ii in range(gsize):
				jj = blocks[j, ii]
				assert jj >= 0
				betas[i,jj] = beta_next[ii]
				# Update beta and mu/r for linear regression
				if betas[i, jj] != 0:
					blas.daxpy(&n, &betas[i, jj], &fT[jj+offset,0], &inc_1, &mu[0], &inc_1)
					if probit == 0: # we have to redo this anyway otherwise
						delta = -1 * betas[i,jj]
						blas.daxpy(&n, &delta, &fT[jj+offset,0], &inc_1, &r[0], &inc_1)

			# This is intentionally outside of the loop resetting jj
			if probit == 1 and (betas[i, jj] != 0 or old_betaj != 0):
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