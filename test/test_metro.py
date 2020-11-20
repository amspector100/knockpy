import time
import os
import pytest
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import treewidth
from scipy import stats
import unittest
from .context import knockpy
from knockpy import utilities, dgp, metro
from knockpy.mrc import mvr_loss


class TestMetroProposal(unittest.TestCase):
    def test_gaussian_likelihood(self):

        X = np.array([0.5, 1, 2, 3])
        mu = 0.5
        var = 0.2

        # Scipy result
        norm_rv = stats.norm(loc=mu, scale=np.sqrt(var))
        sp_result = norm_rv.logpdf(X)

        # Custom result
        custom_result = metro.gaussian_log_likelihood(X, mu, var)
        self.assertTrue(
            np.abs(sp_result - custom_result).sum() < 0.001,
            msg=f"scipy result {sp_result} does not match custom result {custom_result}",
        )

    def test_proposal_covs(self):

        # Fake data
        np.random.seed(110)
        n = 5
        p = 200
        dgprocess = dgp.DGP()
        X, _, _, Q, V = dgprocess.sample_data(method="AR1", rho=0.1, n=n, p=p)

        # Metro sampler, proposal params
        metro_sampler = metro.MetropolizedKnockoffSampler(
            lf=lambda x: np.log(x).sum(),
            X=X,
            mu=np.zeros(p),
            Sigma=V,
            undir_graph=np.abs(Q) > 1e-3,
            S=np.eye(p),
        )

        # Test that proposal likelihood is correct
        mu = np.zeros(2 * p)
        mvn = stats.multivariate_normal(mean=mu, cov=metro_sampler.G)

        # Scipy likelihood
        features = mvn.rvs()
        scipy_likelihood = mvn.logpdf(features)

        # Calculate a new likelihood using the proposal params
        X = features[0:p].reshape(1, -1)
        Xstar = features[p:].reshape(1, -1)

        # Base likelihood for first p variables
        loglike = stats.multivariate_normal(mean=np.zeros(p), cov=V).logpdf(X)

        # Likelihood of jth variable given first j - 1
        prev_proposals = None
        for j in range(p):

            # Test = scipy likelihood at this point
            scipy_likelihood = stats.multivariate_normal(
                mean=np.zeros(p + j), cov=metro_sampler.G[0 : p + j, 0 : p + j]
            ).logpdf(features[0 : p + j])
            self.assertTrue(
                np.abs(loglike - scipy_likelihood) < 0.001,
                f"Proposal likelihood for j={j-1} fails: output {loglike}, expected {scipy_likelihood} (scipy)",
            )

            # Add loglike
            loglike += metro_sampler.q_ll(
                Xjstar=Xstar[:, j], X=X, prev_proposals=prev_proposals
            )
            prev_proposals = Xstar[:, 0 : j + 1]

class TestMetroSample(unittest.TestCase):
    def test_ar1_sample(self):

        # Fake data
        np.random.seed(110)
        n = 30000
        p = 8
        dgprocess = dgp.DGP()
        X, _, _, Q, V = dgprocess.sample_data(method="AR1", n=n, p=p)
        ksampler = knockpy.knockoffs.GaussianSampler(X=X, Sigma=V, method="mvr")
        S = ksampler.fetch_S()

        # Graph structure + junction tree
        Q_graph = np.abs(Q) > 1e-5
        Q_graph = Q_graph - np.eye(p)

        # Metro sampler + likelihood
        mvn = stats.multivariate_normal(mean=np.zeros(p), cov=V)

        def mvn_likelihood(X):
            return mvn.logpdf(X)

        gamma = 0.9999
        metro_sampler = metro.MetropolizedKnockoffSampler(
            lf=mvn_likelihood,
            X=X,
            mu=np.zeros(p),
            Sigma=V,
            undir_graph=Q_graph,
            S=S,
            gamma=gamma,
        )

        # Output knockoffs
        Xk = metro_sampler.sample_knockoffs()

        # Acceptance rate should be exactly one
        acc_rate = metro_sampler.final_acc_probs.mean()
        self.assertTrue(
            acc_rate - gamma > -1e-3,
            msg=f"For AR1 gaussian design, metro has acc_rate={acc_rate} < gamma={gamma}",
        )

        # Check covariance matrix
        features = np.concatenate([X, Xk], axis=1)
        emp_corr_matrix = np.corrcoef(features.T)
        G = np.concatenate(
            [np.concatenate([V, V - S]), np.concatenate([V - S, V]),], axis=1
        )

        np.testing.assert_almost_equal(
            emp_corr_matrix,
            G,
            decimal=2,
            err_msg=f"For AR1 gaussian design, metro does not match theoretical matrix",
        )

    def test_dense_sample(self):

        # Fake data
        np.random.seed(110)
        n = 10000
        p = 4
        dgprocess = dgp.DGP()
        X, _, _, Q, V = dgprocess.sample_data(
            method="blockequi", rho=0.6, n=n, p=p, gamma=1, block_size=p
        )
        ksampler = knockpy.knockoffs.GaussianSampler(X=X, Sigma=V, method="mvr")
        S = ksampler.fetch_S()

        # Network graph
        Q_graph = np.abs(Q) > 1e-5
        Q_graph = Q_graph - np.eye(p)
        undir_graph = nx.Graph(Q_graph)
        width, T = treewidth.treewidth_decomp(undir_graph)
        order, active_frontier = metro.get_ordering(T)

        # Metro sampler and likelihood
        mvn = stats.multivariate_normal(mean=np.zeros(p), cov=V)

        def mvn_likelihood(X):
            return mvn.logpdf(X)

        gamma = 0.99999
        metro_sampler = metro.MetropolizedKnockoffSampler(
            lf=mvn_likelihood,
            X=X,
            mu=np.zeros(p),
            Sigma=V,
            order=order,
            active_frontier=active_frontier,
            gamma=gamma,
            S=S,
            metro_verbose=True,
        )

        # Output knockoffs
        Xk = metro_sampler.sample_knockoffs()

        # Acceptance rate should be exactly one
        acc_rate = metro_sampler.final_acc_probs.mean()
        self.assertTrue(
            acc_rate - gamma > -1e-3,
            msg=f"For equi gaussian design, metro has acc_rate={acc_rate} < gamma={gamma}",
        )

        # Check covariance matrix
        features = np.concatenate([X, Xk], axis=1)
        emp_corr_matrix = np.corrcoef(features.T)
        G = np.concatenate(
            [np.concatenate([V, V - S]), np.concatenate([V - S, V]),], axis=1
        )

        np.testing.assert_almost_equal(
            emp_corr_matrix,
            G,
            decimal=2,
            err_msg=f"For equi gaussian design, metro does not match theoretical matrix",
        )


class TestARTK(unittest.TestCase):
    def test_t_log_likelihood(self):

        # Fake data
        np.random.seed(110)
        n = 15
        p = 10
        df_t = 5
        X1 = np.random.randn(n, p)
        X2 = np.random.randn(n, p)

        # Scipy ratios
        sp_like1 = stats.t.logpdf(X1, df=df_t)
        sp_like2 = stats.t.logpdf(X2, df=df_t)
        sp_diff = sp_like1 - sp_like2

        # Custom ratios
        custom_like1 = metro.t_log_likelihood(X1, df_t=df_t)
        custom_like2 = metro.t_log_likelihood(X2, df_t=df_t)
        custom_diff = custom_like1 - custom_like2

        np.testing.assert_almost_equal(
            custom_diff,
            sp_diff,
            decimal=2,
            err_msg=f"custom t_log_likelihood and scipy t.logpdf disagree",
        )

    def test_tmarkov_likelihood(self):

        # Data
        np.random.seed(110)
        n = 15
        p = 10
        df_t = 5
        X1 = np.random.randn(n, p)
        X2 = np.random.randn(n, p)
        V = np.eye(p)
        Q = np.eye(p)

        # Scipy likelihood ratio for X, scale matrix
        inv_scale = np.sqrt(df_t / (df_t - 2))
        sp_like1 = stats.t.logpdf(inv_scale * X1, df=df_t).sum(axis=1)
        sp_like2 = stats.t.logpdf(inv_scale * X2, df=df_t).sum(axis=1)
        sp_ratio = sp_like1 - sp_like2

        # General likelihood
        rhos = np.zeros(p - 1)
        ar1_like1 = metro.t_markov_loglike(X1, rhos, df_t=df_t)
        ar1_like2 = metro.t_markov_loglike(X2, rhos, df_t=df_t)
        ar1_ratio = ar1_like1 - ar1_like2

        self.assertTrue(
            np.abs(ar1_ratio - sp_ratio).sum() < 0.01,
            f"AR1 ratio {ar1_ratio} and scipy ratio {sp_ratio} disagree for independent t vars",
        )

        # Test again with df_t --> infinity, so it should be approx gaussian
        dgprocess = dgp.DGP()
        X1, _, _, Q, V = dgprocess.sample_data(n=n, p=p, method="AR1", a=3, b=1)
        X2 = np.random.randn(n, p)

        # Ratio using normals
        df_t = 100000
        mu = np.zeros(p)
        norm_like1 = stats.multivariate_normal(mean=mu, cov=V).logpdf(X1)
        norm_like2 = stats.multivariate_normal(mean=mu, cov=V).logpdf(X2)
        norm_ratio = norm_like1 - norm_like2

        # Ratios using T
        rhos = np.diag(V, 1)
        ar1_like1 = metro.t_markov_loglike(X1, rhos, df_t=df_t)
        ar1_like2 = metro.t_markov_loglike(X2, rhos, df_t=df_t)
        ar1_ratio = ar1_like1 - ar1_like2

        self.assertTrue(
            np.abs(ar1_ratio - norm_ratio).mean() < 0.01,
            f"AR1 ratio {ar1_ratio} and gaussian ratio {norm_ratio} disagree for corr. t vars, df={df_t}",
        )

        # Check consistency of tsampler class
        tsampler = metro.ARTKSampler(X=X1, Sigma=V, df_t=df_t,)
        new_ar1_like1 = tsampler.lf(tsampler.X)
        self.assertTrue(
            np.abs(ar1_like1 - new_ar1_like1).sum() < 0.01,
            f"AR1 loglike inconsistent between class ({new_ar1_like1}) and function ({ar1_ratio})",
        )

    def test_tmarkov_samples(self):

        # Test to make sure low df --> heavy tails
        # and therefore acceptances < 1
        np.random.seed(110)
        n = 1000000
        p = 5
        df_t = 3
        dgprocess = dgp.DGP()
        X, _, _, Q, V = dgprocess.sample_data(
            n=n, p=p, method="AR1", rho=0.3, x_dist="ar1t", df_t=df_t
        )
        for S in [None, np.eye(p)]:

            # Sample t
            tsampler = metro.ARTKSampler(X=X, Sigma=V, df_t=df_t, S=S, metro_verbose=True)

            # Correct junction tree
            self.assertTrue(
                tsampler.width == 1, f"tsampler should have width 1, not {tsampler.width}"
            )

            # Sample
            Xk = tsampler.sample_knockoffs()

            # Check empirical means
            # Check empirical covariance matrix
            muk_hat = np.mean(Xk, axis=0)
            np.testing.assert_almost_equal(
                muk_hat,
                np.zeros(p),
                decimal=2,
                err_msg=f"For ARTK sampler, empirical mean of Xk does not match mean of X",
            )

            # Check empirical covariance matrix
            Vk_hat = np.corrcoef(Xk.T)
            np.testing.assert_almost_equal(
                V,
                Vk_hat,
                decimal=2,
                err_msg=f"For ARTK sampler, empirical covariance of Xk does not match cov of X",
            )

            # Check that marginal fourth moments match
            X4th = np.mean(np.power(X, 4), axis=0)
            Xk4th = np.mean(np.power(Xk, 4), axis=0)
            np.testing.assert_almost_equal(
                X4th / 10,
                Xk4th / 10,
                decimal=1,
                err_msg=f"For ARTK sampler, fourth moment of Xk does not match theoretical fourth moment",
            )

            # Run a ton of KS tests
            tsampler.check_xk_validity(X, Xk, testname="ARTK")


class TestBlockT(unittest.TestCase):
    def test_tmvn_log_likelihood(self):

        # Fake data
        np.random.seed(110)
        n = 10
        p = 10
        df_t = 100000

        # Test that the likelihood --> gaussian as df_t --> infinity
        dgprocess = dgp.DGP()
        X1, _, _, Q, V = dgprocess.sample_data(
            n=n, p=p, method="blockequi", gamma=0.3, rho=0.8, x_dist="blockt"
        )
        X2 = np.random.randn(n, p)

        # Ratio using normals
        mu = np.zeros(p)
        norm_like1 = stats.multivariate_normal(mean=mu, cov=V).logpdf(X1)
        norm_like2 = stats.multivariate_normal(mean=mu, cov=V).logpdf(X2)
        norm_ratio = norm_like1 - norm_like2

        # Ratios using T
        tmvn_like1 = metro.t_mvn_loglike(X1, Q, df_t=df_t)
        tmvn_like2 = metro.t_mvn_loglike(X2, Q, df_t=df_t)
        tmvn_ratio = tmvn_like1 - tmvn_like2
        self.assertTrue(
            np.abs(tmvn_ratio - norm_ratio).mean() < 0.01,
            f"T MVN ratio {tmvn_ratio} and gaussian ratio {norm_ratio} disagree for corr. t vars, df={df_t}",
        )

    def test_blockt_samples(self):

        # Test to make sure low df --> heavy tails
        # and therefore acceptances < 1
        np.random.seed(110)
        n = 2000000
        p = 6
        df_t = 5
        dgprocess = dgp.DGP()
        X, _, _, Q, V = dgprocess.sample_data(
            n=n,
            p=p,
            method="blockequi",
            rho=0.4,
            gamma=0,
            block_size=3,
            x_dist="blockt",
            df_t=df_t,
        )
        for S in [np.eye(p), None]:

            # Sample t
            tsampler = metro.BlockTSampler(X=X, Sigma=V, df_t=df_t, S=S, metro_verbose=True)

            # Sample
            Xk = tsampler.sample_knockoffs()

            # Check empirical means
            # Check empirical covariance matrix
            muk_hat = np.mean(Xk, axis=0)
            np.testing.assert_almost_equal(
                muk_hat,
                np.zeros(p),
                decimal=2,
                err_msg=f"For block T sampler, empirical mean of Xk does not match mean of X",
            )

            # Check empirical covariance matrix
            Vk_hat = np.cov(Xk.T)
            np.testing.assert_almost_equal(
                V,
                Vk_hat,
                decimal=2,
                err_msg=f"For block T sampler, empirical covariance of Xk does not match cov of X",
            )

            # Check that marginal fourth moments match
            X4th = np.mean(np.power(X, 4), axis=0)
            Xk4th = np.mean(np.power(Xk, 4), axis=0)
            np.testing.assert_almost_equal(
                X4th / 10,
                Xk4th / 10,
                decimal=1,
                err_msg=f"For block T sampler, fourth moment of Xk does not match theoretical fourth moment",
            )

            # Run a ton of KS tests
            tsampler.check_xk_validity(X, Xk, testname="BLOCKT")


class TestGibbsGraph(unittest.TestCase):
    def test_divconquer_likelihoods(self):

        # Test to make sure the way we split up
        # cliques does not change the likelihood
        np.random.seed(110)
        n = 10
        p = 625
        mu = np.zeros(p)
        dgprocess = dgp.DGP()
        X, _, _, _, _ = dgprocess.sample_data(
            n=n, p=p, method="ising", x_dist="gibbs",
        )
        gibbs_graph = dgprocess.gibbs_graph
        np.fill_diagonal(gibbs_graph, 1)

        # Read V
        file_directory = os.path.dirname(os.path.abspath(__file__))
        V = np.loadtxt(f"{file_directory}/test_covs/vout{p}.txt")

        # Initialize sampler
        metro_sampler = metro.GibbsGridSampler(
            X=X, gibbs_graph=gibbs_graph, mu=mu, Sigma=V, max_width=2,
        )

        # Non-divided likelihood
        nondiv_like = 0
        for clique, lp in zip(metro_sampler.cliques, metro_sampler.log_potentials):
            nondiv_like += lp(X[:, np.array(clique)])

        # Divided likelihood for the many keys
        many_div_like = np.zeros(n)
        for dc_key in metro_sampler.dc_keys:
            # Initialize likelihood for these data points
            div_like = 0
            # Helpful constants
            seps = metro_sampler.separators[dc_key]
            n_inds = metro_sampler.X_ninds[dc_key]
            # Add separator-to-separator cliques manually
            for clique, lp in zip(metro_sampler.cliques, metro_sampler.log_potentials):
                if clique[0] not in seps or clique[1] not in seps:
                    continue
                sepX = X[n_inds]
                div_like += lp(sepX[:, np.array(clique)])

            # Now loop through other blocks
            div_dict_list = metro_sampler.divconq_info[dc_key]
            for block_dict in div_dict_list:
                blockX = X[n_inds][:, block_dict["inds"]]
                for clique, lp in zip(block_dict["cliques"], block_dict["lps"]):
                    div_like += lp(blockX[:, clique])
            many_div_like[n_inds] = np.array(div_like)

        # Test to make sure these likelihoods agree
        np.testing.assert_almost_equal(
            nondiv_like,
            many_div_like,
            decimal=5,
            err_msg=f"Non-divided clique potentials {nondiv_like} do not agree with divided cliques {div_like}",
        )

    def test_large_ising_samples(self):

        # Test that sampling does not throw an error
        np.random.seed(110)
        n = 100
        p = 625
        mu = np.zeros(p)
        dgprocess = dgp.DGP()
        X, _, _, _, _ = dgprocess.sample_data(
            n=n, p=p, method="ising", x_dist="gibbs",
        )
        gibbs_graph = dgprocess.gibbs_graph
        np.fill_diagonal(gibbs_graph, 1)

        # We load custom cov/q matrices for this
        file_directory = os.path.dirname(os.path.abspath(__file__))
        V = np.loadtxt(f"{file_directory}/test_covs/vout{p}.txt")
        Q = np.loadtxt(f"{file_directory}/test_covs/qout{p}.txt")
        max_nonedge = np.max(np.abs(Q[gibbs_graph == 0]))
        self.assertTrue(
            max_nonedge < 1e-5,
            f"Estimated precision for ising{p} has max_nonedge {max_nonedge} >= 1e-5",
        )

        # Initialize sampler
        metro_sampler = metro.GibbsGridSampler(
            X=X,
            gibbs_graph=gibbs_graph,
            mu=mu,
            Sigma=V,
            Q=Q,
            max_width=5,
            method="equicorrelated",
        )

        # Sample and hope for no errors
        Xk = metro_sampler.sample_knockoffs()

    def test_small_ising_samples(self):

        # Test samples to make sure the
        # knockoff properties hold
        np.random.seed(110)
        n = 100000
        p = 9
        mu = np.zeros(p)
        dgprocess = dgp.DGP()
        X, _, _, _, _ = dgprocess.sample_data(
            n=n, p=p, method="ising", x_dist="gibbs",
        )
        gibbs_graph = dgprocess.gibbs_graph
        np.fill_diagonal(gibbs_graph, 1)

        # We load custom cov/q matrices for this
        file_directory = os.path.dirname(os.path.abspath(__file__))
        V = np.loadtxt(f"{file_directory}/test_covs/vout{p}.txt")
        Q = np.loadtxt(f"{file_directory}/test_covs/qout{p}.txt")
        max_nonedge = np.max(np.abs(Q[gibbs_graph == 0]))
        self.assertTrue(
            max_nonedge < 1e-5,
            f"Estimated precision for ising{p} has max_nonedge {max_nonedge} >= 1e-5",
        )

        # Initialize sampler
        metro_sampler = metro.GibbsGridSampler(
            X=X, gibbs_graph=gibbs_graph, mu=mu, Sigma=V, Q=Q, max_width=2,
        )

        # Sample
        Xk = metro_sampler.sample_knockoffs()

        # Check empirical means
        # Check empirical covariance matrix
        mu_hat = X.mean(axis=0)
        muk_hat = np.mean(Xk, axis=0)
        np.testing.assert_almost_equal(
            muk_hat,
            mu_hat,
            decimal=2,
            err_msg=f"For Ising sampler, empirical mean of Xk does not match mean of X",
        )

        # Check empirical covariance matrix
        V_hat = np.cov(X.T)
        Vk_hat = np.cov(Xk.T)
        np.testing.assert_almost_equal(
            V_hat / 2,
            Vk_hat / 2,
            decimal=1,
            err_msg=f"For Ising sampler, empirical covariance of Xk does not match cov of X",
        )

        # Check that marginal fourth moments match
        X4th = np.mean(np.power(X, 4), axis=0)
        Xk4th = np.mean(np.power(Xk, 4), axis=0)
        np.testing.assert_almost_equal(
            X4th / 10,
            Xk4th / 10,
            decimal=1,
            err_msg=f"For Ising sampler, fourth moment of Xk does not match theoretical fourth moment",
        )

        # Run a ton of KS tests
        metro_sampler.check_xk_validity(
            X, Xk, testname="SMALL_ISING",
        )


if __name__ == "__main__":
    unittest.main()
