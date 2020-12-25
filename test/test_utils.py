import numpy as np
import scipy as sp
import unittest
from .context import knockpy

from knockpy import dgp, utilities


def myfunc(a, b, c, d):
    """ A random function. Important to define this globally
	because the multiprocessing package can't pickle locally
	defined functions."""
    return a ** b + c * d


class TestUtils(unittest.TestCase):
    """ Tests some various utility functions """

    def test_group_manipulations(self):
        """ Tests calc_group_sizes and preprocess_groups """

        # Calc group sizes
        groups = np.concatenate([np.ones(2), np.ones(3) + 1, np.ones(2) + 3])
        group_sizes = utilities.calc_group_sizes(groups)
        expected = np.array([2, 3, 0, 2])
        np.testing.assert_array_almost_equal(
            group_sizes,
            expected,
            decimal=6,
            err_msg="Incorrectly calculates group sizes",
        )

        # Make sure this raises the correct error for groups
        # which include zero
        groups = np.arange(0, 10, 1)
        self.assertRaisesRegex(
            ValueError, "groups cannot contain 0", utilities.calc_group_sizes, groups,
        )

        # Make sure this raises the correct error for groups
        # which include zero
        groups = [1.5, 1.5, 1.8, 20.4]
        self.assertRaisesRegex(
            TypeError,
            "groups cannot contain non-integer values",
            utilities.calc_group_sizes,
            groups,
        )

        # Preprocess
        groups = np.array([0.3, 0.24, 0.355, 0.423, 0.423, 0.3])
        processed_groups = utilities.preprocess_groups(groups)
        expected = np.array([2, 1, 3, 4, 4, 2])
        np.testing.assert_array_almost_equal(
            processed_groups,
            expected,
            decimal=6,
            err_msg="Incorrectly preprocesses groups",
        )

        # Test function which calculates selections
        non_nulls = [-1, 0, -1, 1, 0, 0, 0]
        groups = [1, 1, 2, 2, 3, 3, 3]
        out = utilities.fetch_group_nonnulls(non_nulls, groups)
        expected = np.array([1.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(
            out, expected, decimal=6, err_msg="Incorrectly calculates group selections"
        )

    def test_random_permutation(self):
        """ Tests random permutation """

        # Calculate random permutation, see if rev_inds correctly undoes inds
        test_list = np.array([0, 5, 3, 6, 32, 2, 1])
        inds, rev_inds = utilities.random_permutation_inds(len(test_list))
        reconstructed = test_list[inds][rev_inds]
        np.testing.assert_array_almost_equal(
            test_list,
            reconstructed,
            decimal=6,
            err_msg="Random permutation is not correctly reversed",
        )

    def test_force_pos_def(self):

        # Random symmetric matrix, will have highly neg eigs
        np.random.seed(110)
        X = np.random.randn(100, 100)
        X = (X.T + X) / 2

        # Force pos definite
        tol = 1e-3
        posX = utilities.shift_until_PSD(X, tol=tol)
        mineig = np.linalg.eigh(posX)[0].min()

        # Make sure the difference between the tolerance and is small
        self.assertTrue(
            mineig >= tol - 1e-5,  # Acct for num. errors in eigval calc
            msg="Minimum eigenvalue is not greater than or equal to tolerance",
        )

    def test_chol2inv(self):

        # Random pos def matrix
        X = np.random.randn(100, 100)
        X = np.dot(X.T, X)

        # Check cholesky decomposition
        inverse = utilities.chol2inv(X)
        np.testing.assert_array_almost_equal(
            np.eye(100),
            np.dot(X, inverse),
            decimal=6,
            err_msg="chol2inv fails to correctly calculate inverses",
        )

    def test_misaligned_covariance_estimation(self):

        # Inputs
        seed = 110
        sample_kwargs = {
            "n": 640,
            "p": 300,
            "method": "blockequi",
            "gamma": 1,
            "rho": 0.8,
        }

        # Extracta couple of constants
        n = sample_kwargs["n"]
        p = sample_kwargs["p"]

        # Create data generating process
        np.random.seed(seed)
        dgprocess = dgp.DGP()
        X, y, beta, _, V = dgprocess.sample_data(**sample_kwargs)

        # Make sure this does not raise an error
        # (even though it is ill-conditioned and the graph lasso doesn't fail)
        utilities.estimate_covariance(X, shrinkage="graphicallasso")

    def test_covariance_estimation(self):

        # Random data
        np.random.seed(110)
        n = 50
        p = 100
        rho = 0.3
        V = (1 - rho) * np.eye(p) + (rho) * np.ones((p, p))
        dgprocess = dgp.DGP(Sigma=V)
        X, _, _, _, _ = dgprocess.sample_data(n=n)

        # Estimate covariance matrix
        Vest, _ = utilities.estimate_covariance(X, tol=1e-2)
        frobenius = np.sqrt(np.power(Vest - V, 2).mean())
        self.assertTrue(
            frobenius < 0.2, f"High-dimension covariance estimation is horrible"
        )

        # Test factor approximation, should be quite good
        D, U = utilities.estimate_factor(Vest, num_factors=5)
        V_factor = np.diag(D) + np.dot(U, U.T)
        frobenius = np.sqrt(np.power(V_factor - Vest, 2).mean())
        target = 0.1
        self.assertTrue(
            frobenius < target, f"Factor approximation is very poor (frob error={frobenius} > {target})"
        )


    def test_apply_pool(self):

        # Apply_pool for num_processes = 1
        a_vals = [1, 2, 3, 4]
        b_vals = [1, 2, 3, 4]
        c_vals = [1, 3, 5, 7]
        d = 100
        out = utilities.apply_pool(
            func=myfunc,
            a=a_vals,
            b=b_vals,
            c=c_vals,
            constant_inputs={"d": d},
            num_processes=1,
        )
        expected = myfunc(a_vals[0], b_vals[0], c_vals[0], 100)
        self.assertEqual(
            out[0],
            expected,
            f"Apply_pool yields incorrect answer ({out[0]} vs {expected}",
        )

        # Try again for num_processes = 4
        out2 = utilities.apply_pool(
            func=myfunc,
            a=a_vals,
            b=b_vals,
            c=c_vals,
            constant_inputs={"d": d},
            num_processes=4,
        )
        self.assertEqual(
            out, out2, f"Apply_pool yields different answers for different processes"
        )

    def test_blockdiag_to_blocks(self):

        # Create block sizes and blocks
        block_nos = utilities.preprocess_groups(np.random.randint(1, 50, 100))
        block_nos = np.sort(block_nos)
        block_sizes = utilities.calc_group_sizes(block_nos)
        blocks = [np.random.randn(b, b) for b in block_sizes]

        # Create block diagonal matrix in scipy
        block_diag = sp.linalg.block_diag(*blocks)
        blocks2 = utilities.blockdiag_to_blocks(block_diag, block_nos)
        for expected, out in zip(blocks, blocks2):
            np.testing.assert_almost_equal(
                out,
                expected,
                err_msg="blockdiag_to_blocks incorrectly separates blocks",
            )


if __name__ == "__main__":
    unittest.main()
