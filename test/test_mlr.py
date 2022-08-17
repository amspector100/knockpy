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
			update_p0=False,
			update_sigma2=0,
			n_iter=2000,
			chains=20,
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
		print("p0s:")
		print(fxmlr.p0s.mean(axis=0))
		print(mxmlr.p0s.mean())
		print("sigma2s:")
		print(fxmlr.sigma2s.mean())
		print(mxmlr.sigma2s.mean())
		print("tau2s:")
		print(fxmlr.tau2s.mean(axis=0))
		print(mxmlr.tau2s.mean())

		print(np.around(W1, 2))
		print(np.around(W2, 2))
		raise ValueError()