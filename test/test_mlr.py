import unittest

import numpy as np

import knockpy
from knockpy import knockoffs, mlr, utilities


class TestMLR(unittest.TestCase):
    def test_calc_group_blocks(self):
        p = 30
        for j in range(5):
            # Create random groups
            groups = np.random.choice(np.arange(p), p, replace=True)
            groups = utilities.preprocess_groups(groups)
            group_sizes = utilities.calc_group_sizes(groups)
            max_gsize = np.max(group_sizes)
            # create blocks
            blocks = mlr.mlr._calc_group_blocks(groups, group_sizes)
            groups -= 1  # makes indexing easier
            for gj in range(group_sizes.shape[0]):
                expected = np.where(groups == gj)[0]
                # Make dimensions line up
                ne = expected.shape[0]
                if ne < max_gsize:
                    expected = np.concatenate(
                        [expected, -1 * np.ones(max_gsize - ne)], axis=0
                    )
                # Test equality
                np.testing.assert_array_almost_equal(
                    expected,
                    blocks[gj],
                    decimal=6,
                    err_msg=f"for groups={groups}, block for group {gj} is incorrect",
                )

    # def test_fx_equals_mx(self):
    # 	"""
    # 	Check MX and FX give the same results for FX knockoffs.
    # 	"""
    # 	# Create FX knockoffs
    # 	np.random.seed(111)
    # 	n_iter = 1000
    # 	chains = 10
    # 	n = 200
    # 	p = 20
    # 	X = np.random.randn(n, p)
    # 	X = X - X.mean(axis=0)
    # 	beta = np.random.randn(p)
    # 	y = X @ beta + np.random.randn(n)
    # 	ksampler = knockpy.knockoffs.FXSampler(X=X)
    # 	mlr_kwargs = dict(
    # 		p0_a0=1.0,
    # 		p0_b0=1.0,
    # 		tau2_a0=2.0,
    # 		tau2_b0=1.0,
    # 		sigma2=1.0,
    # 		update_p0=True,
    # 		update_sigma2=False,
    # 		n_iter=n_iter,
    # 		chains=chains,
    # 	)

    # 	# 1. Fit FX MLR spikeslab
    # 	fxmlr = mlr.MLR_FX_Spikeslab(
    # 		num_mixture=1, **copy.deepcopy(mlr_kwargs)
    # 	)
    # 	kf1 = KF(ksampler=ksampler, fstat=fxmlr)
    # 	kf1.forward(X=X, y=y)
    # 	W1 = kf1.W

    # 	# 2. Fit MX MLR spikeslab
    # 	mxmlr = mlr.MLR_Spikeslab(
    # 		min_p0=0.0, p0=0.1, **copy.deepcopy(mlr_kwargs)
    # 	)
    # 	kf2 = KF(ksampler=ksampler, fstat=mxmlr)
    # 	kf2.forward(X=X, y=y)
    # 	W2 = kf2.W

    # 	# params are roughly equal
    # 	params = ['p0', 'tau2', 'sigma2']
    # 	fxests = [
    # 		fxmlr.p0s.mean(axis=0)[0],
    # 		fxmlr.tau2s.mean(axis=0)[0],
    # 		fxmlr.sigma2s.mean()
    # 	]
    # 	mxests = [
    # 		mxmlr.p0s.mean(),
    # 		mxmlr.tau2s.mean(),
    # 		mxmlr.sigma2s.mean(0)
    # 	]
    # 	for pname, fxest, mxest in zip(
    # 		params, fxests, mxests
    # 	):
    # 		np.testing.assert_almost_equal(
    # 			fxest,
    # 			mxest,
    # 			decimal=1,
    # 			err_msg=f"Estimated {pname} for FX and MX disagree."
    # 		)

    # 	# W statistics
    # 	print("FX W:", np.around(W1, 2))
    # 	print("MX W:", np.around(W2, 2))
    # 	np.testing.assert_almost_equal(
    # 		np.abs(W1 - W2).mean(),
    # 		0.0,
    # 		decimal=1,
    # 		err_msg=f"||W_FX - W_MX||_1 >= 0.1"
    # 	)

    def test_mlr_oracle(self):
        # Data generating process
        np.random.seed(111)
        n_iter = 2000
        chains = 10
        n = 65
        p = 30
        X = np.random.randn(n, p)
        X = X - X.mean(axis=0)
        beta = np.random.randn(p) / 10
        y = X @ beta + np.random.randn(n)

        # knockoffs
        ksampler = knockoffs.FXSampler(X=X)
        Xk = ksampler.sample_knockoffs()

        # Expected oracle log-probs
        S = X.T @ X - X.T @ Xk
        Sinv = np.diag(1 / np.diag(S))
        Delta = np.diag(2 * Sinv)
        tildebeta = np.dot(Sinv, np.dot(X.T - Xk.T, y))
        expected_log_probs = 2 * np.abs(tildebeta * beta) / Delta

        # Fit oracle statistic
        oracle = mlr.OracleMLR(beta=beta, sigma2=1.0, n_iter=n_iter, chains=chains)
        oracle.fit(X=X, Xk=Xk, y=y, groups=None)
        np.testing.assert_almost_equal(
            np.abs(oracle.etas.mean(axis=0)),
            expected_log_probs,
            decimal=2,
            err_msg="OracleMLR statistic log probs do not agree with theoretical values from Whiteout",
        )

        # repeat for probit, make sure no errors
        oracle_logistic = mlr.OracleMLR(beta=beta, n_iter=200, chains=3)
        oracle_logistic.fit(X=X, Xk=Xk, y=(y > 0).astype(float), groups=None)

    def test_group_mlr_no_errors(self):
        """
        To do: more rigorous tests would be nice
        """

        # Create FX knockoffs
        np.random.seed(111)
        n_iter = 10
        chains = 1
        n = 200
        p = 20
        X = np.random.randn(n, p)
        X = X - X.mean(axis=0)
        beta = np.random.randn(p)
        y = X @ beta + np.random.randn(n)
        ksampler = knockpy.knockoffs.GaussianSampler(X=X)
        Xk = ksampler.sample_knockoffs()
        mlr_kwargs = dict(
            n_iter=n_iter,
            chains=chains,
        )

        # Create MLR fstat
        groups = np.random.choice(np.arange(int(p / 2)), p, replace=True)
        groups = utilities.preprocess_groups(groups)
        mxmlr = mlr.MLR_Spikeslab(min_p0=0.0, p0=0.5, **mlr_kwargs)
        mxmlr.fit(X=X, Xk=Xk, y=y, groups=groups)

        # Do again for MLR splines
        splinemlr = mlr.MLR_Spikeslab_Splines(**mlr_kwargs, degree=3, knots=5)
        splinemlr.fit(X=X, Xk=Xk, y=y, groups=groups, n_iter=200)
