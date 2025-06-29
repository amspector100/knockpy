import unittest

import numpy as np
import pytest

from knockpy import ggm


class TestGGM(unittest.TestCase):
    """
    Tests FDR control for gaussian
    graphical models.
    """

    def test_discovered_edges(self):
        # Random W and T
        np.random.seed(111)
        p = 500
        W = np.random.randn(p, p)
        W -= np.diag(np.diag(W))
        T = np.random.randn(p)
        T[5] = np.inf
        # Check results
        for logic in ["and", "or"]:
            edges = ggm.discovered_edges(W=W, T=T, logic=logic)
            for i in range(p):
                for j in range(p):
                    if edges[i, j]:
                        if logic == "and":
                            self.assertTrue((W[i, j] >= T[i]) & (W[j, i] >= T[j]))
                        else:
                            self.assertTrue((W[i, j] >= T[i]) | (W[j, i] >= T[j]))
                    else:
                        if logic == "and":
                            self.assertTrue((W[i, j] < T[i]) | (W[j, i] < T[j]))
                        else:
                            self.assertTrue((W[i, j] < T[i]) & (W[j, i] < T[j]))

    def test_ggm_fdr(self):
        # Create W obeying flip-sign by column
        reps, p, q = 256, 200, 0.1
        # set of true edges
        true_edges = np.random.binomial(1, 0.1, (p, p)).astype(bool)
        true_edges = true_edges | true_edges.T
        for j in range(p):
            true_edges[j, j] = False

        # W statistics under the global null and a non-global null
        for tedges in [true_edges, np.zeros((p, p)).astype(bool)]:
            for logic in ["and", "or"]:
                fdrs = []
                for _ in range(reps):
                    # Create W-statistics
                    W = np.random.randn(p, p)
                    W[tedges] += 3
                    W -= np.diag(np.diag(W))
                    # find threshold
                    T = ggm.compute_ggm_threshold(W, fdr=q, logic=logic)
                    dedges = ggm.discovered_edges(W=W, T=T, logic=logic)
                    fdr = (dedges & (~tedges)).sum() / max(1.0, dedges.sum())
                    fdrs.append(fdr)
                # This method is conservative so it is safe to assert
                # that the empirical FDR is lower than the nominal FDR
                fdr = np.mean(fdrs)
                self.assertTrue(
                    fdr <= q,
                    f"For gaussian graphical model, realized fdr={fdr} > target level q={q}.",
                )

    def test_ggm_filter(self):
        """Make sure knockoff GGM does not error"""
        n, p = 100, 10
        X = np.random.randn(n, p)
        gkf = ggm.KnockoffGGM()
        gkf.forward(X=X, fdr=0.1, ggm_kwargs=dict(a=0.01))


if __name__ == "__main__":
    import sys

    import pytest

    pytest.main(sys.argv)
    # unittest.main()
