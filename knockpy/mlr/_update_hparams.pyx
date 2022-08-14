# cython: profile=False

cimport cython
import numpy as np
import scipy.stats
cimport numpy as np
import scipy.linalg
cimport scipy.linalg.cython_blas as blas


# Blas commonly used parameters
cdef int inc_1 = 1

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef int _update_hparams(
    # Constants
    int i,
    int n,
    int p,
    int max_nprop,
    # Inputs
    double[:, ::1] betas, #double[::1] betai,
    double[::1] r,
    double[::1] p0s,
    double[::1] sigma2s,
    double[::1] tau2s,
    double invgamma,
    # Previously allocated memory
    double[::1] p0_proposals,
    # Prior on hyperparameters
    int update_tau2,
	double tau2_a0,
	double tau2_b0,
	int update_sigma2,
	double sigma2_a0,
	double sigma2_b0,
	int update_p0,
	double min_p0,
	double p0_a0,
	double p0_b0,
):
    # Definitions
    cdef:
        int num_active = 0
        double sample_var = 0
        double sigma_b = 0

    # Calculate number of active variables
    for j in range(p):
        if betas[i,j] != 0:
            num_active += 1

    # Resample p0s
    if update_p0 == 1:
        # sample p0
        if min_p0 == 0:
            p0s[i] = np.random.beta(
                p0_a0 + p - num_active, p0_b0 + num_active
            )
        else:
            # rejection sampling
            p0_proposals = np.random.beta(
                p0_a0 + p - num_active, p0_b0 + num_active, size=max_nprop
            ) # batching the proposals is more efficient 
            p0s[i] = min_p0
            for j in range(max_nprop):
                if p0_proposals[j] > min_p0:
                    p0s[i] = p0_proposals[j]
                    break

    # possibly resample sigma2
    if update_sigma2 == 1:
        # calculate l2 norm of r
        r2 = blas.dnrm2(&n, &r[0], &inc_1)
        r2 = r2 * r2
        # compute b parameter and rescale
        sigma_b = r2 / 2.0 + sigma2_b0
        sigma2s[i] = sigma_b * invgamma

    # possibly resample tau2
    if update_tau2:
        sample_var = 0
        for j in range(p):
            if betas[i,j] != 0:
                sample_var += betas[i,j] * betas[i,j]
        tau2s[i] = (tau2_b0 + sample_var / 2.0) / np.random.gamma(
            shape=tau2_a0 + float(num_active) / 2.0
        )

    return 0
