import copy
import time
import pytest
import numpy as np
from scipy import stats
import unittest
from .context import knockpy
from .context import file_directory

from knockpy import knockoffs, mlr, utilities
from knockpy.knockoff_filter import KnockoffFilter as KF


class TestMLR(unittest.TestCase):

	def test_fx_equals_mx(self):
		"""
		Check MX and FX give the same results for FX knockoffs.
		"""
		# Create FX knockoffs
		np.random.seed(111)
		n_iter = 1000
		chains = 10
		n = 200
		p = 20
		X = np.random.randn(n, p)
		X = X - X.mean(axis=0)
		beta = np.random.randn(p)
		y = X @ beta + np.random.randn(n)
		ksampler = knockpy.knockoffs.FXSampler(X=X)
		mlr_kwargs = dict(
			p0_a0=1.0,
			p0_b0=1.0,
			tau2_a0=2.0,
			tau2_b0=1.0,
			sigma2=1.0,
			update_p0=True,
			update_sigma2=False,
			n_iter=n_iter,
			chains=chains,
		)

		# 1. Fit FX MLR spikeslab
		fxmlr = mlr.MLR_FX_Spikeslab(
			num_mixture=1, **copy.deepcopy(mlr_kwargs)
		)
		kf1 = KF(ksampler=ksampler, fstat=fxmlr)
		kf1.forward(X=X, y=y)
		W1 = kf1.W

		# 2. Fit MX MLR spikeslab
		mxmlr = mlr.MLR_MX_Spikeslab(
			min_p0=0.0, p0=0.5,  **copy.deepcopy(mlr_kwargs)
		)
		kf2 = KF(ksampler=ksampler, fstat=mxmlr)
		kf2.forward(X=X, y=y)
		W2 = kf2.W

		# params are roughly equal
		params = ['p0', 'tau2', 'sigma2']
		fxests = [
			fxmlr.p0s.mean(axis=0)[0], 
			fxmlr.tau2s.mean(axis=0)[0],
			fxmlr.sigma2s.mean()
		]
		mxests = [
			mxmlr.p0s.mean(),
			mxmlr.tau2s.mean(),
			mxmlr.sigma2s.mean(0)
		]
		for pname, fxest, mxest in zip(
			params, fxests, mxests
		):
			np.testing.assert_almost_equal(
				fxest, 
				mxest, 
				decimal=1,
				err_msg=f"Estimated {pname} for FX and MX disagree."
			)

		# W statistics
		print("FX W:", np.around(W1, 2))
		print("MX W:", np.around(W2, 2))
		np.testing.assert_almost_equal(
			np.abs(W1 - W2).mean(),
			0.0,
			decimal=1,
			err_msg=f"||W_FX - W_MX||_1 >= 0.1"
		)